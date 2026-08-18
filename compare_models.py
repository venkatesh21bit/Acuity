import argparse
import os
import glob
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# Import model, transforms, and metrics
from models.nafnet import build_model
from data.transforms import LogVST
from utils.metrics import MetricsAggregator

def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    raw_state = state.get("model", state)
    model.load_state_dict(raw_state, strict=True)

def load_image_as_tensor(path: str) -> torch.Tensor:
    """Load an image and return as a (1, 1, H, W) float32 tensor in [0, 1]."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".npy":
        img = np.load(path).astype(np.float32)
    else:
        img = np.array(Image.open(path).convert('L')).astype(np.float32) / 255.0
    
    if img.ndim == 3:
        img = img[..., 0]
        
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description="Compare multiple NAFNet models on a subset of images.")
    parser.add_argument("--models_dir", default="checkpoints", help="Directory containing .pth models")
    parser.add_argument("--lr_dir",     default="dataset/train/train/NoisyLR", help="Path to Noisy images (LR)")
    parser.add_argument("--gt_dir",     default="dataset/train/train/GT", help="Path to Ground Truth images (GT)")
    parser.add_argument("--num_images", type=int, default=50, help="Number of images to use for testing")
    parser.add_argument("--output_dir", default="model_comparisons", help="Where to save restored samples (optional)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 1. Find all models
    model_paths = glob.glob(os.path.join(args.models_dir, "*.pth"))
    if not model_paths:
        print(f"❌ No .pth files found in {args.models_dir}!")
        return
    print(f"📦 Found {len(model_paths)} models to compare.")

    # 2. Get paired images
    valid_exts = {".npy", ".png", ".jpg"}
    lr_files = sorted([f for f in os.listdir(args.lr_dir) if os.path.splitext(f)[-1].lower() in valid_exts])
    
    # Pick a subset of images
    lr_files = lr_files[:args.num_images]
    image_pairs = []
    
    for f in lr_files:
        lr_path = os.path.join(args.lr_dir, f)
        # Assuming GT files have the exact same name
        gt_path = os.path.join(args.gt_dir, f)
        if os.path.exists(gt_path):
            image_pairs.append((lr_path, gt_path))
            
    print(f"🖼️  Selected {len(image_pairs)} image pairs for evaluation.")

    # 3. Setup Model, LogVST, and Metrics Aggregator
    log_vst = LogVST(eps=1e-3).to(device)
    model_cfg = {
        "channels": 1,
        "model": {
            "width": 64,
            "enc_blocks": [2, 2, 4, 8],
            "dec_blocks": [2, 2, 2, 2],
            "upscale": 2
        },
        "train": {"compile": False}
    }
    model = build_model(model_cfg, device)
    
    # We will compute PSNR, SSIM, LPIPS using the provided utils
    metrics_agg = MetricsAggregator(lpips_device=str(device))
    
    results = []

    # 4. Loop through each model
    for ckpt_path in model_paths:
        model_name = Path(ckpt_path).stem
        print(f"\n[{model_name}] Loading weights...")
        
        try:
            load_checkpoint(model, ckpt_path, device)
        except Exception as e:
            print(f"⚠️ Failed to load {model_name}: {e}")
            continue
            
        model.eval()
        metrics_agg.reset()
        
        # Save one sample image per model for visual comparison
        sample_saved = False

        with torch.no_grad():
            for lr_path, gt_path in tqdm(image_pairs, desc=f"Evaluating {model_name}"):
                # Load tensors
                lr_tensor = load_image_as_tensor(lr_path).to(device)
                gt_tensor = load_image_as_tensor(gt_path).to(device)
                
                # Pre-process
                lr_vst = log_vst(lr_tensor)
                
                # Inference (Float32)
                pred = model(lr_vst)
                
                # Update metrics
                metrics_agg.update(pred.cpu(), gt_tensor.cpu())
                
                # Save first image for visual comparison
                if not sample_saved:
                    pred_arr = pred.cpu().squeeze().numpy()
                    img_uint8 = (pred_arr * 255.0).clip(0, 255).astype(np.uint8)
                    Image.fromarray(img_uint8).save(os.path.join(args.output_dir, f"sample_{model_name}.png"))
                    sample_saved = True

        # Store results
        m = metrics_agg.compute()
        results.append({
            "name": model_name,
            "psnr": m["psnr"],
            "ssim": m["ssim"],
            "lpips": m["lpips"]
        })

    # 5. Print Summary Table
    print("\n" + "="*70)
    print(f"{'Model Name':<25} | {'PSNR (dB) ↑':<12} | {'SSIM ↑':<10} | {'LPIPS ↓':<10}")
    print("-" * 70)
    
    for res in results:
        print(f"{res['name']:<25} | {res['psnr']:<12.3f} | {res['ssim']:<10.4f} | {res['lpips']:<10.4f}")
    
    print("="*70)
    
    # 6. Find the best model
    if results:
        best_psnr = max(results, key=lambda x: x["psnr"])
        best_lpips = min(results, key=lambda x: x["lpips"])
        
        print(f"\n🏆 Best PSNR:  {best_psnr['name']} ({best_psnr['psnr']:.3f} dB)")
        print(f"🏆 Best LPIPS: {best_lpips['name']} ({best_lpips['lpips']:.4f})")
        
        if best_psnr['name'] == best_lpips['name']:
            print(f"\n👑 {best_psnr['name']} is the undisputed best model mathematically!")
        else:
            print("\n⚖️ The math is split. Check the generated samples in the output folder to decide which one looks better to your human eyes!")

if __name__ == "__main__":
    main()
