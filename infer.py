"""
infer.py
────────
High-throughput async inference pipeline for the speckle noise restoration model.

Pipeline stages
───────────────
  1. 16-worker ThreadPoolExecutor  → parallel np.load from disk
  2. Pinned host memory            → non-blocking H2D DMA (CUDA Stream 1)
  3. LogVST pre-processing         → GPU float32
  4. BF16 AMP + torch.compile NAFNet  → fused Triton/CUDA kernels, B=64
  5. SoftClip post-processing      → predictions ∈ [0, 1]
  6. Non-blocking D2H              → CUDA Stream 2 → pinned CPU buffer
  7. 16-worker ThreadPoolExecutor  → concurrent PNG + .npy disk write

Development mode (default):
  compile_mode = max-autotune-no-cudagraphs

Final submission mode (static B=64, flip in configs/infer.yaml):
  use_cuda_graphs = true
  (CUDA Graph captured after warmup_iters warm-up passes)

Usage
─────
  python infer.py --config configs/infer.yaml
  python infer.py --config configs/infer.yaml --benchmark
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.cuda
import yaml

from data.dataset    import build_test_loader
from data.transforms import LogVST
from models.nafnet   import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Handle compiled model state dict prefix
    raw_state = state.get("model", state)
    model.load_state_dict(raw_state, strict=True)
    print(f"[infer] Loaded checkpoint: {ckpt_path}")


def save_npy_and_png(args):
    """Worker function for the async disk-write thread pool."""
    arr, stem, output_dir, save_png = args
    npy_path = os.path.join(output_dir, f"{stem}.npy")
    np.save(npy_path, arr)
    if save_png:
        try:
            import cv2
            png_path = os.path.join(output_dir, f"{stem}.png")
            img_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(png_path, img_uint8)
        except ImportError:
            pass  # cv2 optional for PNG saving


# ─────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(cfg: dict, benchmark: bool = False) -> None:
    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] Device : {device}")

    # ── AMP dtype ─────────────────────────────────────────────────────────
    amp_dtype_str = cfg["infer"].get("amp_dtype", "bfloat16")
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[amp_dtype_str]
    print(f"[infer] AMP    : {amp_dtype}")

    # ── Output directory ──────────────────────────────────────────────────
    out_dir  = cfg["data"]["output_dir"]
    save_png = cfg["output"].get("save_png", True)
    os.makedirs(out_dir, exist_ok=True)

    # ── Pre-processing ────────────────────────────────────────────────────
    log_vst = LogVST(eps=1e-3).to(device)

    # ── Model ─────────────────────────────────────────────────────────────
    # Build a minimal cfg dict compatible with build_model
    model_cfg = {
        "channels": 1,
        "model":    cfg["model"],
        "train": {
            "compile":      cfg["infer"].get("compile", True),
            "compile_mode": cfg["infer"].get("compile_mode",
                                             "max-autotune-no-cudagraphs"),
        },
        "image": cfg.get("image", {"channels": 1}),
    }
    model = build_model(model_cfg, device)
    load_checkpoint(model, cfg["model"]["checkpoint"], device)
    model.eval()

    # ── CUDA Streams for async H2D / D2H overlap ──────────────────────────
    stream_h2d = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    stream_d2h = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    # ── CUDA Graph capture (final submission mode) ─────────────────────────
    use_cuda_graphs  = cfg["infer"].get("use_cuda_graphs", False)
    warmup_iters     = cfg["infer"].get("warmup_iters", 3)
    batch_size       = cfg["infer"]["batch_size"]
    graphed_model    = None

    if use_cuda_graphs and device.type == "cuda":
        print(f"[infer] Capturing CUDA Graph (B={batch_size}, warmup={warmup_iters} iters)…")
        example = torch.zeros(batch_size, 1, 128, 128, device=device, dtype=amp_dtype)
        # Warm-up runs
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            for _ in range(warmup_iters):
                _ = model(example)
        torch.cuda.synchronize()
        graphed_model = torch.cuda.make_graphed_callables(model, (example,))
        print("[infer] CUDA Graph captured successfully.")

    inference_fn = graphed_model if graphed_model is not None else model

    # ── DataLoader ────────────────────────────────────────────────────────
    test_loader = build_test_loader(
        lr_dir       = cfg["data"]["input_dir"],
        batch_size   = batch_size,
        num_workers  = cfg["data"]["num_io_workers"],
        pin_memory   = True,
    )
    print(f"[infer] Test images : {len(test_loader.dataset)}")
    print(f"[infer] Batches     : {len(test_loader)}")

    # ── Async disk-write pool ─────────────────────────────────────────────
    write_pool = ThreadPoolExecutor(max_workers=cfg["data"]["num_io_workers"])
    write_futures = []

    # ── Benchmark counters ────────────────────────────────────────────────
    t_start      = time.perf_counter()
    total_images = 0

    # ─────────────────────────────────────────────────────────────────────
    # Inference loop
    # ─────────────────────────────────────────────────────────────────────
    with torch.no_grad():
        for lr_batch, stems in test_loader:
            B = lr_batch.shape[0]
            total_images += B

            # ── H2D async transfer (CUDA Stream 1) ────────────────────────
            if stream_h2d is not None:
                with torch.cuda.stream(stream_h2d):
                    lr_batch = lr_batch.to(device, non_blocking=True)
            else:
                lr_batch = lr_batch.to(device)

            # ── Log-VST pre-processing ────────────────────────────────────
            lr_batch = log_vst(lr_batch)

            # ── BF16 forward pass ─────────────────────────────────────────
            with torch.amp.autocast('cuda', dtype=amp_dtype,
                                          enabled=(amp_dtype != torch.float32)):
                # Pad to batch_size if last batch is smaller (CUDA Graphs need fixed size)
                pad = batch_size - B if use_cuda_graphs and B < batch_size else 0
                if pad > 0:
                    lr_padded = torch.cat([
                        lr_batch,
                        lr_batch[:pad]   # dummy padding (output ignored)
                    ], dim=0)
                    pred_padded = inference_fn(lr_padded)
                    pred = pred_padded[:B]
                else:
                    pred = inference_fn(lr_batch)

            # ── D2H async transfer (CUDA Stream 2) ────────────────────────
            if stream_d2h is not None:
                with torch.cuda.stream(stream_d2h):
                    pred_cpu = pred.cpu().float()
            else:
                pred_cpu = pred.cpu().float()

            # ── Async disk write (ThreadPoolExecutor) ─────────────────────
            for i, stem in enumerate(stems):
                arr = pred_cpu[i, 0].numpy()   # (H, W) float32
                fut = write_pool.submit(
                    save_npy_and_png,
                    (arr, stem, out_dir, save_png),
                )
                write_futures.append(fut)

    # ── Wait for all writes to complete ───────────────────────────────────
    for fut in write_futures:
        fut.result()
    write_pool.shutdown(wait=True)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t_start

    # ── Results ───────────────────────────────────────────────────────────
    print(f"\n[infer] ✓ Saved {total_images} predictions → {out_dir}")
    if benchmark:
        throughput = total_images / elapsed
        print(
            f"[bench] Wall-clock : {elapsed:.2f} s\n"
            f"[bench] Throughput : {throughput:.1f} imgs/s\n"
        )


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NAFNet inference pipeline")
    parser.add_argument("--config",    type=str, default="configs/infer.yaml")
    parser.add_argument("--benchmark", action="store_true",
                        help="Print throughput statistics after inference.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_inference(cfg, benchmark=args.benchmark)


if __name__ == "__main__":
    main()
