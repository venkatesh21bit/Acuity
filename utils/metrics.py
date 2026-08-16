"""
utils/metrics.py
────────────────
Evaluation metrics for the denoising / super-resolution pipeline.

Metrics computed
────────────────
  • PSNR  – Peak Signal-to-Noise Ratio  (dB, higher is better)
  • SSIM  – Structural Similarity Index (0–1, higher is better)
  • LPIPS – Learned Perceptual Image Patch Similarity (lower is better)

All functions operate on float32 tensors in [0, 1] on the CPU.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from typing import Optional

try:
    from pytorch_msssim import ssim as _ssim_fn
    _SSIM_AVAILABLE = True
except ImportError:
    _SSIM_AVAILABLE = False

try:
    import lpips as lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# PSNR
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(
    pred:   torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        pred       : (B, C, H, W) float32 ∈ [0, 1]
        target     : (B, C, H, W) float32 ∈ [0, 1]
        data_range : Maximum pixel value. Default 1.0.

    Returns:
        psnr_db (float): Mean PSNR over the batch in dB.
    """
    mse = F.mse_loss(pred, target, reduction="mean").item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(data_range ** 2 / mse)


# ─────────────────────────────────────────────────────────────────────────────
# SSIM
# ─────────────────────────────────────────────────────────────────────────────

def compute_ssim(
    pred:   torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """
    Compute Structural Similarity Index Metric (SSIM).

    Args:
        pred       : (B, C, H, W) float32 ∈ [0, 1]
        target     : (B, C, H, W) float32 ∈ [0, 1]
        data_range : Maximum pixel value. Default 1.0.

    Returns:
        ssim_val (float): Mean SSIM ∈ [0, 1].
    """
    if not _SSIM_AVAILABLE:
        raise RuntimeError("pytorch-msssim is required for SSIM. Install with: pip install pytorch-msssim")
    return _ssim_fn(pred, target, data_range=data_range, size_average=True).item()


# ─────────────────────────────────────────────────────────────────────────────
# LPIPS
# ─────────────────────────────────────────────────────────────────────────────

class LPIPSEvaluator:
    """
    Singleton wrapper around the LPIPS VGG-16 evaluator.

    Loads the LPIPS model once and reuses it across calls to avoid redundant
    network initialisation during the eval loop.

    Usage:
        evaluator = LPIPSEvaluator(device="cuda")
        score = evaluator(pred, target)  # float
    """

    def __init__(self, net: str = "vgg", device: str = "cpu"):
        if not _LPIPS_AVAILABLE:
            raise RuntimeError("lpips is required. Install with: pip install lpips")
        self.model = lpips_lib.LPIPS(net=net).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """
        Args:
            pred   : (B, C, H, W) float32 ∈ [0, 1]
            target : (B, C, H, W) float32 ∈ [0, 1]

        Returns:
            lpips_val (float): Mean LPIPS distance (lower is better).
        """
        # Scale to [-1, 1] as expected by LPIPS
        p = (pred   * 2.0 - 1.0).to(self.device)
        t = (target * 2.0 - 1.0).to(self.device)

        if p.shape[1] == 1:
            p = p.expand(-1, 3, -1, -1)
            t = t.expand(-1, 3, -1, -1)

        return self.model(p, t).mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics table
# ─────────────────────────────────────────────────────────────────────────────

class MetricsAggregator:
    """
    Accumulates per-batch metrics and reports a formatted aggregate summary.

    Usage:
        agg = MetricsAggregator()
        for pred, gt in val_loop:
            agg.update(pred.cpu(), gt.cpu())
        agg.report()
    """

    def __init__(self, lpips_device: str = "cpu"):
        self._psnr_sum  = 0.0
        self._ssim_sum  = 0.0
        self._lpips_sum = 0.0
        self._count     = 0
        self._lpips_eval: Optional[LPIPSEvaluator] = None
        self._lpips_device = lpips_device

    def _get_lpips(self) -> LPIPSEvaluator:
        if self._lpips_eval is None:
            self._lpips_eval = LPIPSEvaluator(device=self._lpips_device)
        return self._lpips_eval

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Add one batch of predictions and targets.

        Args:
            pred   : (B, 1, H, W) float32 ∈ [0, 1]
            target : (B, 1, H, W) float32 ∈ [0, 1]
        """
        B = pred.shape[0]
        self._psnr_sum  += compute_psnr(pred, target)  * B
        self._ssim_sum  += compute_ssim(pred, target)  * B
        self._lpips_sum += self._get_lpips()(pred, target) * B
        self._count     += B

    def compute(self) -> dict:
        """Return dict of mean metrics over all accumulated batches."""
        if self._count == 0:
            return {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        return {
            "psnr":  self._psnr_sum  / self._count,
            "ssim":  self._ssim_sum  / self._count,
            "lpips": self._lpips_sum / self._count,
        }

    def reset(self) -> None:
        self._psnr_sum  = 0.0
        self._ssim_sum  = 0.0
        self._lpips_sum = 0.0
        self._count     = 0

    def report(self) -> None:
        m = self.compute()
        print(
            f"\n{'─'*45}\n"
            f"  PSNR  : {m['psnr']:>8.4f} dB   (↑ higher is better)\n"
            f"  SSIM  : {m['ssim']:>8.4f}       (↑ higher is better)\n"
            f"  LPIPS : {m['lpips']:>8.4f}       (↓ lower  is better)\n"
            f"{'─'*45}"
        )
