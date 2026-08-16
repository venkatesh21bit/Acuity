"""
losses/compound_loss.py
───────────────────────
Compound multi-objective loss for joint denoising and super-resolution.

Components
──────────
  1. CharbonnierLoss   – smooth L1 variant; maximises PSNR without over-smoothing.
  2. MSSSIMLoss        – multi-scale structural similarity; preserves contrast / structure.
  3. LPIPSLoss         – deep feature distance (VGG-16); reconstructs perceptual textures.
  4. SobelGradientLoss – penalises edge blurring; boosts SSIM / sharpness.

Multi-Objective Optimisation (homoscedastic uncertainty weighting)
──────────────────────────────────────────────────────────────────
  One learnable log-variance parameter per loss component:

      L_total = Σ_k  [ (1 / (2 σ_k²)) · L_k  +  log σ_k ]

  where  σ_k = exp(log_sigma_k).  The log_sigma_k tensors are added to the
  optimiser parameter group alongside model weights so they are updated
  automatically during standard backpropagation — zero extra backward passes.

  Reference: Kendall & Gal, "Multi-Task Learning Using Uncertainty to
             Weigh Losses for Scene Geometry and Semantics", CVPR 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# pytorch-msssim provides a differentiable MS-SSIM implementation
try:
    from pytorch_msssim import ms_ssim
    _MSSSIM_AVAILABLE = True
except ImportError:
    _MSSSIM_AVAILABLE = False
    print("[WARNING] pytorch-msssim not found. MSSSIMLoss will fall back to SSIM.")
    from pytorch_msssim import ssim

# lpips provides a VGG-16–backed perceptual distance
try:
    import lpips as lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False
    print("[WARNING] lpips not found. LPIPSLoss will be disabled (weight set to 0).")


# ─────────────────────────────────────────────────────────────────────────────
# Individual loss components
# ─────────────────────────────────────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    """
    Smooth approximation to L1 (Charbonnier / pseudo-Huber):

        L_charb(y, ŷ) = mean( sqrt( (y − ŷ)² + ε² ) )

    Penalises large residuals (edges) more heavily than MSE while remaining
    smooth near zero — preserves structural sharpness during training.

    Args:
        eps (float): Numerical stability offset. Default 1e-3.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class MSSSIMLoss(nn.Module):
    """
    Multi-Scale Structural Similarity loss:

        L_msssim = 1 − MS-SSIM(y, ŷ)

    Evaluates luminance, contrast, and structure at multiple spatial scales,
    penalising halo artifacts introduced by deblurring filters.

    Requires pytorch-msssim. Falls back to single-scale SSIM if unavailable.

    Args:
        data_range (float): Pixel value range. Default 1.0 (float32 images).
    """

    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # MS-SSIM requires at least 160×160 for 5-scale; use SSIM fallback for 256×256 patches
        if _MSSSIM_AVAILABLE:
            score = ms_ssim(pred, target, data_range=self.data_range, size_average=True)
        else:
            score = ssim(pred, target, data_range=self.data_range, size_average=True)
        return 1.0 - score


class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity loss (VGG-16 backbone).

    Minimising LPIPS forces the network to reconstruct high-frequency visual
    textures that match the perceptual distribution of clean ground-truth images.

    The VGG model is loaded once and frozen; its weights are NOT trained.

    Args:
        net (str): Backbone network. 'vgg' (default) or 'alex'.
    """

    def __init__(self, net: str = "vgg"):
        super().__init__()
        if _LPIPS_AVAILABLE:
            self.loss_fn = lpips_lib.LPIPS(net=net)
            # Freeze LPIPS weights — they are not part of the training objective
            for p in self.loss_fn.parameters():
                p.requires_grad_(False)
        else:
            self.loss_fn = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_fn is None:
            return pred.new_zeros(1).squeeze()

        # LPIPS expects inputs in [-1, 1]; our tensors are in [0, 1]
        pred_scaled   = pred   * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0

        # LPIPS requires 3-channel input; broadcast grayscale to RGB
        if pred.shape[1] == 1:
            pred_scaled   = pred_scaled.expand(-1, 3, -1, -1)
            target_scaled = target_scaled.expand(-1, 3, -1, -1)

        return self.loss_fn(pred_scaled, target_scaled).mean()


class SobelGradientLoss(nn.Module):
    """
    Spatial gradient loss using Sobel edge operators:

        L_sobel = mean( |Sobel_x(y − ŷ)| + |Sobel_y(y − ŷ)| )

    Penalises high-frequency edge degradation and blurring without
    introducing non-differentiable overhead at inference.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels (fixed, not learnable)
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        # (out_ch, in_ch, kH, kW)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target                   # (B, 1, H, W)
        # Apply Sobel filters independently to each batch item
        gx = F.conv2d(diff, self.kx, padding=1)
        gy = F.conv2d(diff, self.ky, padding=1)
        return torch.mean(torch.abs(gx) + torch.abs(gy))


# ─────────────────────────────────────────────────────────────────────────────
# Compound loss with homoscedastic MOO weighting
# ─────────────────────────────────────────────────────────────────────────────

class CompoundLoss(nn.Module):
    """
    Compound multi-objective loss with homoscedastic uncertainty weighting.

    The total loss:

        L_total = Σ_k  [ exp(-log_var_k) · L_k  +  0.5 · log_var_k ]

    where  log_var_k = log(σ_k²)  is a learnable parameter per loss term.
    This formulation:
      • Up-weights certain losses (small σ) and down-weights others automatically.
      • Adds a regularisation term (0.5 · log σ²) preventing σ → ∞ collapse.
      • Requires ZERO extra backward passes compared to GradNorm.

    The log_var parameters are included in the optimiser via
    `model.parameters()` once CompoundLoss is moved onto the device and
    its parameters added to the optimiser group.

    Args:
        lambda_charb   (float): Initial Charbonnier weight (overridden by MOO).
        lambda_msssim  (float): Initial MS-SSIM weight.
        lambda_lpips   (float): Initial LPIPS weight.
        lambda_sobel   (float): Initial Sobel gradient weight.
        charb_eps      (float): Charbonnier ε. Default 1e-3.
    """

    NUM_LOSSES = 4

    def __init__(
        self,
        lambda_charb:  float = 1.0,
        lambda_msssim: float = 0.1,
        lambda_lpips:  float = 0.05,
        lambda_sobel:  float = 0.01,
        charb_eps:     float = 1e-3,
    ):
        super().__init__()

        # Individual loss modules
        self.charb  = CharbonnierLoss(eps=charb_eps)
        self.msssim = MSSSIMLoss(data_range=1.0)
        self.lpips  = LPIPSLoss(net="vgg")
        self.sobel  = SobelGradientLoss()

        # Store initial static weights (used if MOO is disabled)
        self.static_weights = [
            lambda_charb, lambda_msssim, lambda_lpips, lambda_sobel
        ]

        # Homoscedastic uncertainty: one log(σ²) per loss component
        # Initialised to 0 → σ² = 1 → equal initial weighting
        self.log_vars = nn.Parameter(torch.zeros(self.NUM_LOSSES))

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred   : Predicted HR image  (B, 1, 256, 256) ∈ [0, 1]
            target : Ground-truth HR image (B, 1, 256, 256) ∈ [0, 1]

        Returns:
            total_loss (Tensor): scalar loss for .backward()
            loss_dict  (dict):   individual loss values for logging
        """
        losses = [
            self.charb(pred, target),
            self.msssim(pred, target),
            self.lpips(pred, target),
            self.sobel(pred, target),
        ]

        total = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        for k, L in enumerate(losses):
            # L_total_k = exp(-log_var) * L_k  +  0.5 * log_var
            clamped_log_var = torch.clamp(self.log_vars[k], min=-10.0, max=10.0)
            precision = torch.exp(-clamped_log_var)
            total = total + precision * L + 0.5 * clamped_log_var

        loss_dict = {
            "charb":   losses[0].item(),
            "msssim":  losses[1].item(),
            "lpips":   losses[2].item(),
            "sobel":   losses[3].item(),
            "total":   total.item(),
            "sigma_charb":  torch.exp(0.5 * self.log_vars[0]).item(),
            "sigma_msssim": torch.exp(0.5 * self.log_vars[1]).item(),
            "sigma_lpips":  torch.exp(0.5 * self.log_vars[2]).item(),
            "sigma_sobel":  torch.exp(0.5 * self.log_vars[3]).item(),
        }
        return total.squeeze(), loss_dict
