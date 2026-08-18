import argparse
import os
import glob
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# Import the model and pre-processing
from models.nafnet import build_model
from data.transforms import LogVST

def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    raw_state = state.get("model", state)
    model.load_state_dict(raw_state, strict=True)
    print(f"✅ Loaded checkpoint: {ckpt_path}")

def load_image(path: str) -> np.ndarray:
    """Load an image from .npy or standard image formats as a float32 array in [0, 1]."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".npy":
        img = np.load(path).astype(np.float32)
    else:
        # Load with PIL for .png, .jpg, etc.
        img = np.array(Image.open(path).convert('L')).astype(np.float32) / 255.0
    
    # Ensure it's 2D (H, W)
    if img.ndim == 3:
        img = img[..., 0]
    return img

def save_image(arr: np.ndarray, path: str):
    """Save a float32 array as an image."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".npy":
        np.save(path, arr)
    else:
        img_uint8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(path)

def main():
    parser = argparse.ArgumentParser(description="Standalone Evaluation Script for NAFNet")
    parser.add_argument("-i", "--input",  required=True, help="Path to directory containing test images (.npy, .png, etc.)")
    parser.add_argument("-o", "--output", required=True, help="Path to directory to save restored outputs")
    parser.add_argument("-c", "--ckpt",   default="checkpoint/best_psnr.pth", help="Path to model checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 1. Initialize Logarithmic Variance Stabilizing Transformation
    log_vst = LogVST(eps=1e-3).to(device)

    # 2. Build the NAFNet Model
    # Hardcoded configuration to match the trained architecture
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
    
    # 3. Load weights
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    # 4. Find all images in the input directory
    valid_exts = {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    image_paths = []
    for root, _, files in os.walk(args.input):
        for file in files:
            if os.path.splitext(file)[-1].lower() in valid_exts:
                image_paths.append(os.path.join(root, file))
                
    if not image_paths:
        print(f"❌ No valid images found in {args.input}!")
        return

    print(f"🔍 Found {len(image_paths)} images. Starting inference...")

    # 5. Inference Loop
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Evaluating"):
            # Load and format
            img = load_image(path)
            
            # Convert to Tensor (B, C, H, W)
            tensor_img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
            
            # Apply Pre-processing
            tensor_img = log_vst(tensor_img)
            
            # Forward Pass (Float32 for safety during inference)
            pred = model(tensor_img)
            
            # Format Output
            pred_arr = pred.cpu().squeeze().numpy()  # (H, W) float32
            
            # Save Output
            stem = Path(path).stem
            # Save as PNG by default for easy viewing, or match original if you prefer
            # We'll save as PNG so the user can look at the results immediately
            out_path = os.path.join(args.output, f"{stem}_restored.png")
            save_image(pred_arr, out_path)

    print(f"✅ Evaluation complete! All outputs saved to: {args.output}")

if __name__ == "__main__":
    main()
