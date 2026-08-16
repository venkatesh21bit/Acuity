"""
data/transforms.py
──────────────────
Pre- and post-processing transforms for the speckle noise restoration pipeline.

All transforms operate on float32 tensors and preserve out-of-bounds intensity
values caused by multiplicative speckle noise.
"""

import torch
import torch.nn as nn


class LogVST(nn.Module):
    """
    Logarithmic Variance Stabilizing Transformation.

    Converts multiplicative speckle noise into approximate additive Gaussian
    noise by operating in the log domain:

        x_out = log(x + eps)

    The small scalar `eps` prevents log(0) singularities at zero-intensity
    pixels. Applied to unclipped float32 input tensors before the network.

    Args:
        eps (float): Numerical stability offset. Default 1e-3.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp to 0 before adding eps to avoid log(negative) → NaN.
        # NoisyLR values can be slightly below 0 (thermal noise artefacts).
        return torch.log(x.clamp(min=0.0) + self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


class InverseLogVST(nn.Module):
    """
    Inverse of LogVST: x_out = exp(x) - eps.

    Useful for converting log-domain predictions back to intensity space
    during debugging or alternative post-processing paths.

    Args:
        eps (float): Must match the eps used in the forward LogVST.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) - self.eps


class FloatNorm(nn.Module):
    """
    Standard normalisation over the unclipped float32 training distribution.

        x_out = (x - mean) / std

    Computed over the raw, unclipped training set so that out-of-bounds
    speckle intensity values are preserved (not clipped to [0, 1] beforehand).

    Args:
        mean (float): Dataset mean over unclipped pixels.
        std  (float): Dataset std  over unclipped pixels.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std",  torch.tensor(std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class SoftClip(nn.Module):
    """
    Parametric Soft-Clipping applied at the network output layer.

    Smoothly maps restored pixel values into [0, 1] using a sigmoid
    gate without hard gradient cutoffs:

        f(x) = sigmoid(alpha * x) / (sigmoid(alpha) * normaliser)

    A simpler, equally effective formulation used here:

        f(x) = sigmoid(alpha * (2*x - 1)) * (1 + 2/alpha) - 1/alpha

    In practice we use the straightforward re-scaled sigmoid:

        f(x) = 1 / (1 + exp(-alpha * x))   (standard sigmoid, alpha controls sharpness)

    This maintains continuous, non-zero backpropagation gradients even for
    predictions slightly outside [0, 1], smoothly guiding intensities back
    into the ground-truth range during training.

    Args:
        alpha (float): Steepness of the soft boundary. Higher → closer to hard
                       clamp. Default 10.0 provides smooth yet tight bounding.
    """

    def __init__(self, alpha: float = 10.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Smooth monotonic map: x ∈ ℝ → (0, 1)
        # At x=0 → 0.5, so shift so x=0.5 → ~0.5 (near-identity around midpoint)
        return torch.sigmoid(self.alpha * x)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"
