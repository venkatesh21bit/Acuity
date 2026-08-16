"""
train.py
────────
Training loop for the NAFNet denoising / super-resolution pipeline.

Key features
────────────
  • BF16 Automatic Mixed Precision  – identical exponent range to FP32;
    safe for unclipped speckle intensities; no GradScaler needed.
  • Homoscedastic MOO loss          – CompoundLoss.log_vars trained jointly.
  • AdamW + cosine LR schedule with linear warmup.
  • torch.compile(fullgraph=True)   – fused Triton kernels via Inductor.
  • Best-PSNR and best-LPIPS checkpoints saved independently.

Usage
─────
  python train.py --config configs/train.yaml
  python train.py --config configs/train.yaml --overfit-batch 1
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from data.dataset   import build_dataloaders
from data.transforms import LogVST
from losses.compound_loss import CompoundLoss
from models.nafnet  import build_model
from utils.metrics  import MetricsAggregator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(
    state: dict,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  ✓ checkpoint saved → {path}")


def build_optimizer(model: nn.Module, criterion: CompoundLoss, cfg: dict):
    """
    Build AdamW optimizer with two parameter groups:
      1. Model parameters   – with weight decay
      2. MOO log_vars       – no weight decay (these are scale parameters)
    """
    return torch.optim.AdamW(
        [
            {"params": model.parameters(),         "weight_decay": cfg["train"]["weight_decay"]},
            {"params": criterion.log_vars,         "weight_decay": 0.0,   "lr": cfg["train"]["lr"] * 0.1},
        ],
        lr=cfg["train"]["lr"],
    )


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    total_steps  = cfg["train"]["epochs"] * steps_per_epoch
    warmup_steps = cfg["train"]["warmup_steps"]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    criterion: CompoundLoss,
    loader,
    optimizer,
    scheduler,
    scaler:    torch.cuda.amp.GradScaler,
    log_vst:   LogVST,
    device:    torch.device,
    epoch:     int,
    cfg:       dict,
    amp_dtype: torch.dtype,
) -> dict:
    model.train()
    criterion.train()

    log_interval = cfg["logging"]["log_interval"]
    grad_clip    = cfg["train"]["grad_clip"]

    total_loss = 0.0
    step = 0
    t0 = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
    for lr_imgs, gt_imgs in pbar:
        # ── Host → Device (non-blocking DMA via pinned memory) ────────────
        lr_imgs = lr_imgs.to(device, non_blocking=True)   # (B, 1, 128, 128)
        gt_imgs = gt_imgs.to(device, non_blocking=True)   # (B, 1, 256, 256)

        # ── Logarithmic variance stabilisation ────────────────────────────
        lr_imgs = log_vst(lr_imgs)

        # ── Forward under BF16 AMP ────────────────────────────────────────
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            pred = model(lr_imgs)                          # (B, 1, 256, 256)
            loss, loss_dict = criterion(pred, gt_imgs)

        # ── Backward + gradient clip + step ───────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + [criterion.log_vars],
            max_norm=grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss_dict["total"]
        step += 1

        if step % log_interval == 0:
            lr_now = scheduler.get_last_lr()[0]
            pbar.set_postfix(
                loss=f"{loss_dict['total']:.4f}",
                charb=f"{loss_dict['charb']:.4f}",
                lpips=f"{loss_dict['lpips']:.4f}",
                lr=f"{lr_now:.2e}",
            )

    elapsed = time.time() - t0
    return {"avg_loss": total_loss / max(1, step), "time_s": elapsed}


# ─────────────────────────────────────────────────────────────────────────────
# Validation epoch
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:    nn.Module,
    loader,
    log_vst:  LogVST,
    device:   torch.device,
    amp_dtype: torch.dtype,
) -> dict:
    model.eval()
    agg = MetricsAggregator(lpips_device=str(device))

    for lr_imgs, gt_imgs in tqdm(loader, desc="  [val]", leave=False):
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        gt_imgs = gt_imgs.to(device, non_blocking=True)

        lr_imgs = log_vst(lr_imgs)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            pred = model(lr_imgs)

        agg.update(pred.cpu().float(), gt_imgs.cpu().float())

    return agg.compute()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    cfg = load_config(args.config)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # ── AMP dtype ─────────────────────────────────────────────────────────
    amp_dtype_str = cfg["train"].get("amp_dtype", "bfloat16")
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[amp_dtype_str]
    print(f"[train] AMP dtype: {amp_dtype}")

    # ── Transforms ────────────────────────────────────────────────────────
    log_vst = LogVST(eps=1e-3).to(device)

    # ── Data ──────────────────────────────────────────────────────────────
    if args.overfit_batch:
        # Quick sanity check: overfit a single batch
        from data.dataset import SpeckleDataset
        from torch.utils.data import DataLoader
        ds = SpeckleDataset(
            cfg["data"]["train_gt_dir"],
            cfg["data"]["train_lr_dir"],
            indices=list(range(cfg["train"]["batch_size"])),
        )
        train_loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                                  shuffle=False, pin_memory=True, num_workers=0)
        val_loader   = train_loader
    else:
        train_loader, val_loader = build_dataloaders(
            gt_dir       = cfg["data"]["train_gt_dir"],
            lr_dir       = cfg["data"]["train_lr_dir"],
            batch_size   = cfg["train"]["batch_size"],
            val_split    = cfg["data"]["val_split"],
            num_workers  = cfg["data"]["num_workers"],
            pin_memory   = cfg["data"]["pin_memory"],
            prefetch_factor = cfg["data"]["prefetch_factor"],
        )

    print(f"[train] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(cfg, device)

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = CompoundLoss(
        lambda_charb   = cfg["loss"]["lambda_charb"],
        lambda_msssim  = cfg["loss"]["lambda_msssim"],
        lambda_lpips   = cfg["loss"]["lambda_lpips"],
        lambda_sobel   = cfg["loss"]["lambda_sobel"],
        charb_eps      = cfg["loss"]["charb_eps"],
    ).to(device)

    # ── Optimiser + Scheduler ─────────────────────────────────────────────
    optimizer = build_optimizer(model, criterion, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    # ── Checkpoint dir ────────────────────────────────────────────────────
    ckpt_dir = cfg["logging"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
    best_psnr  = -float("inf")
    best_lpips =  float("inf")

    # ─────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler, scaler,
            log_vst, device, epoch, cfg, amp_dtype,
        )

        if epoch % cfg["logging"]["val_interval"] == 0:
            val_metrics = validate(model, val_loader, log_vst, device, amp_dtype)

            psnr  = val_metrics["psnr"]
            ssim  = val_metrics["ssim"]
            lpips = val_metrics["lpips"]
            loss  = train_stats["avg_loss"]

            print(
                f"Epoch {epoch:03d} | "
                f"loss={loss:.4f} | "
                f"PSNR={psnr:.3f} dB | "
                f"SSIM={ssim:.4f} | "
                f"LPIPS={lpips:.4f} | "
                f"time={train_stats['time_s']:.0f}s"
            )

            # ── Save best PSNR checkpoint ──────────────────────────────────
            if psnr > best_psnr:
                best_psnr = psnr
                if cfg["logging"]["save_best_psnr"]:
                    save_checkpoint(
                        {"epoch": epoch, "model": model.state_dict(),
                         "psnr": psnr, "ssim": ssim, "lpips": lpips},
                        os.path.join(ckpt_dir, "best_psnr.pth"),
                    )

            # ── Save best LPIPS checkpoint ─────────────────────────────────
            if lpips < best_lpips:
                best_lpips = lpips
                if cfg["logging"]["save_best_lpips"]:
                    save_checkpoint(
                        {"epoch": epoch, "model": model.state_dict(),
                         "psnr": psnr, "ssim": ssim, "lpips": lpips},
                        os.path.join(ckpt_dir, "best_lpips.pth"),
                    )

    print(f"\n[train] Done.  Best PSNR: {best_psnr:.3f} dB | Best LPIPS: {best_lpips:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        type=str, default="configs/train.yaml")
    parser.add_argument("--overfit-batch", action="store_true",
                        help="Overfit a single batch for sanity checking.")
    main(parser.parse_args())
