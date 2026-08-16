"""
utils/visualize.py
──────────────────
Visualisation utilities for the denoising / super-resolution pipeline.

Functions
─────────
  show_triplet  – side-by-side: NoisyLR (bicubic up) | Prediction | GT
  show_residual – absolute difference map with colour scaling
  save_grid     – save a grid of images to disk as PNG
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")           # headless-safe backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert (1, H, W) or (H, W) float32 tensor → (H, W) numpy uint8 array."""
    if t.ndim == 3:
        t = t.squeeze(0)
    arr = t.detach().cpu().float().clamp(0, 1).numpy()
    return (arr * 255).astype(np.uint8)


def show_triplet(
    lr:   torch.Tensor,
    pred: torch.Tensor,
    gt:   torch.Tensor,
    title: str = "",
    save_path: str | None = None,
) -> None:
    """
    Display / save a side-by-side comparison:
      [Noisy LR (bicubic ×2)] | [Model Prediction] | [Ground Truth]

    Args:
        lr        : (1, 128, 128) float32 — degraded low-resolution input.
        pred      : (1, 256, 256) float32 — restored prediction ∈ [0, 1].
        gt        : (1, 256, 256) float32 — clean ground truth ∈ [0, 1].
        title     : Optional super-title string.
        save_path : If provided, saves the figure instead of displaying it.
    """
    if not _MPL_AVAILABLE:
        raise RuntimeError("matplotlib is required for visualisation.")

    # Bicubic upscale LR to GT resolution for fair visual comparison
    lr_up = F.interpolate(lr.unsqueeze(0), size=gt.shape[-2:], mode="bicubic",
                          align_corners=False).squeeze(0)

    lr_np   = _to_numpy(lr_up)
    pred_np = _to_numpy(pred)
    gt_np   = _to_numpy(gt)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, name in zip(
        axes,
        [lr_np, pred_np, gt_np],
        ["NoisyLR (bicubic ×2)", "Model Prediction", "Ground Truth"],
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=11)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def show_residual(
    pred:   torch.Tensor,
    gt:     torch.Tensor,
    title:  str = "Absolute Residual",
    save_path: str | None = None,
    scale:  float = 5.0,
) -> None:
    """
    Display an amplified absolute difference map between prediction and GT.

    Bright regions indicate reconstruction errors. Useful for identifying
    structured artefacts (e.g. halo, checkerboard, noise residuals).

    Args:
        pred       : (1, H, W) float32 ∈ [0, 1]
        gt         : (1, H, W) float32 ∈ [0, 1]
        title      : Figure title.
        save_path  : Save path (PNG). If None, displays interactively.
        scale      : Amplification factor for the residual. Default 5×.
    """
    if not _MPL_AVAILABLE:
        raise RuntimeError("matplotlib is required for visualisation.")

    diff = (pred - gt).abs().clamp(0, 1) * scale
    diff_np = _to_numpy(diff)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(diff_np, cmap="hot", vmin=0, vmax=255)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def save_grid(
    images:    list[torch.Tensor],
    save_path: str,
    nrow:      int = 8,
    pad:       int = 2,
) -> None:
    """
    Arrange a list of (1, H, W) tensors into a grid and save as PNG.

    Args:
        images    : List of float32 tensors ∈ [0, 1].
        save_path : Output PNG path.
        nrow      : Images per row.
        pad       : Padding pixels between images.
    """
    if not _MPL_AVAILABLE:
        raise RuntimeError("matplotlib is required for visualisation.")

    n   = len(images)
    ncol = nrow
    nrows = math.ceil(n / ncol)
    H, W = images[0].shape[-2], images[0].shape[-1]

    # Build canvas
    canvas = np.full(
        ((H + pad) * nrows + pad, (W + pad) * ncol + pad),
        128, dtype=np.uint8
    )
    for i, img in enumerate(images):
        r, c = divmod(i, ncol)
        y0 = pad + r * (H + pad)
        x0 = pad + c * (W + pad)
        canvas[y0:y0 + H, x0:x0 + W] = _to_numpy(img)

    fig, ax = plt.subplots(1, 1, figsize=(ncol * 1.5, nrows * 1.5))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
