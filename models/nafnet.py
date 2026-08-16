"""
models/nafnet.py
────────────────
Nonlinear Activation-Free Network (NAFNet) adapted for ×2 super-resolution
with multiplicative speckle and Gaussian noise removal.

Architecture overview
─────────────────────
  Input (B, 1, 128, 128)  [unclipped float32, after LogVST]
       │
  [Stem: Conv 3×3]
       │
  ┌────┴──────────────────────────────────┐
  │  Encoder – 4 levels                   │
  │  NAFBlock×N + stride-2 Conv downsample│
  └────┬──────────────────────────────────┘
       │  Bottleneck NAFBlocks
  ┌────┴──────────────────────────────────┐
  │  Decoder – 4 levels                   │
  │  stride-2 ConvT upsample + skip concat│
  │  + NAFBlock×N                         │
  └────┬──────────────────────────────────┘
       │
  [PixelShufflePack  r=2]  ← feature extraction at LR resolution
       │
  [SoftClip α=10]          ← smooth [0,1] output boundary
       │
  Output (B, 1, 256, 256)  ∈ (0, 1)

Key design choices (from report)
─────────────────────────────────
  • SimpleGate  – replaces ReLU/GELU; halves channels via elementwise product.
  • SCA         – global avg-pool + 1×1 Conv (no Sigmoid); allows weights > 1.
  • LayerNorm   – per-sample stable under small batch sizes.
  • PixelShuffle as the sole spatial upsampler at the final layer only.
  • torch.compile(mode='max-autotune-no-cudagraphs', fullgraph=True).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.transforms import SoftClip


# ─────────────────────────────────────────────────────────────────────────────
# Primitive building blocks
# ─────────────────────────────────────────────────────────────────────────────

class SimpleGate(nn.Module):
    """
    Gated activation without nonlinear transcendental operations.

    Splits the channel dimension into two halves and returns their
    elementwise product:

        SG(x) = x[:, :C/2] ⊙ x[:, C/2:]

    Mimics gated activations (e.g. GLU) while reducing FLOPs and
    memory access cost versus ReLU / GELU.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """
    Simplified Channel Attention (SCA).

    Replaces Squeeze-and-Excitation's MLP + Sigmoid pair with a single
    linear 1×1 convolution operating on a global average-pooled summary.
    Omitting Sigmoid allows attention weights to exceed unity or go negative,
    expanding dynamic feature calibration.

    Args:
        channels (int): Number of input (and output) channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.pool   = nn.AdaptiveAvgPool2d(1)          # (B, C, 1, 1)
        self.conv1x1 = nn.Conv2d(channels, channels, 1) # linear; no bias needed here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.pool(x)
        attn = self.conv1x1(attn)
        return x * attn


# ─────────────────────────────────────────────────────────────────────────────
# Core NAFBlock
# ─────────────────────────────────────────────────────────────────────────────

class NAFBlock(nn.Module):
    """
    Nonlinear Activation-Free Block.

    Processing order:
      1. LayerNorm  (per-sample; stable at any batch size)
      2. 1×1 Conv   (expand channels ×2 for SimpleGate)
      3. DW 3×3 Conv (local spatial mixing)
      4. SimpleGate  (halves channels back)
      5. SCA         (channel recalibration)
      6. 1×1 Conv    (project back to C channels)
      7. Residual add

    The second branch (FFN-style) also applies LayerNorm → 1×1 → SimpleGate
    before merging with a second scaling factor.

    Args:
        channels   (int): Base channel width C.
        ffn_expand (float): Expansion ratio for the FFN branch. Default 2.0.
    """

    def __init__(self, channels: int, ffn_expand: float = 2.0):
        super().__init__()
        dw_ch  = channels * 2          # after first expansion (for SimpleGate → C)
        ffn_ch = int(channels * ffn_expand * 2)  # after FFN expansion

        # ── Spatial mixing branch ──────────────────────────────────────────
        self.norm1   = nn.LayerNorm(channels)
        self.conv1   = nn.Conv2d(channels, dw_ch, 1, bias=True)
        self.dw_conv = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch, bias=True)
        self.gate1   = SimpleGate()                  # dw_ch → channels
        self.sca     = SimplifiedChannelAttention(channels)
        self.conv2   = nn.Conv2d(channels, channels, 1, bias=True)

        # ── FFN branch ────────────────────────────────────────────────────
        self.norm2   = nn.LayerNorm(channels)
        self.ffn1    = nn.Conv2d(channels, ffn_ch, 1, bias=True)
        self.gate2   = SimpleGate()                  # ffn_ch → ffn_ch//2
        self.ffn2    = nn.Conv2d(ffn_ch // 2, channels, 1, bias=True)

        # Learnable per-block scaling (initialised to small values)
        self.beta  = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-3)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── Spatial mixing branch ──────────────────────────────────────────
        shortcut = x
        # LayerNorm applied channel-wise: permute to (B, H, W, C)
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.dw_conv(x)
        x = self.gate1(x)          # (B, C, H, W)
        x = self.sca(x)
        x = self.conv2(x)
        x = shortcut + x * self.beta

        # ── FFN branch ────────────────────────────────────────────────────
        shortcut = x
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.ffn1(x)
        x = self.gate2(x)          # (B, ffn_ch//2, H, W)
        x = self.ffn2(x)
        x = shortcut + x * self.gamma

        return x


# ─────────────────────────────────────────────────────────────────────────────
# Sub-pixel convolution upsampler (PixelShuffle)
# ─────────────────────────────────────────────────────────────────────────────

class PixelShufflePack(nn.Module):
    """
    Sub-pixel convolution upsampling head.

    Feature extraction stays at the lower-resolution spatial domain; spatial
    upscaling is applied only at this final layer, minimising HBM3 bandwidth.

        Conv(in_ch, in_ch × r², 3, padding=1) → PixelShuffle(r)

    Args:
        in_ch   (int): Input channel count.
        out_ch  (int): Output channel count after PixelShuffle.
        scale   (int): Upscale factor r (e.g. 2).
    """

    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.conv  = nn.Conv2d(in_ch, out_ch * scale * scale, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return F.pixel_shuffle(x, self.scale)


# ─────────────────────────────────────────────────────────────────────────────
# Full NAFNet model
# ─────────────────────────────────────────────────────────────────────────────

class NAFNet(nn.Module):
    """
    U-shaped NAFNet for ×2 joint denoising and super-resolution.

    Args:
        in_ch      (int): Input channels (1 for grayscale).
        width      (int): Base channel width. Default 64.
        enc_blocks (list[int]): Number of NAFBlocks at each encoder level.
        dec_blocks (list[int]): Number of NAFBlocks at each decoder level.
        upscale    (int): PixelShuffle upscaling factor. Default 2.
        softclip_alpha (float): Alpha for the output SoftClip layer.
    """

    def __init__(
        self,
        in_ch:          int        = 1,
        width:          int        = 64,
        enc_blocks:     list[int]  = (2, 2, 4, 8),
        dec_blocks:     list[int]  = (2, 2, 2, 2),
        upscale:        int        = 2,
        softclip_alpha: float      = 10.0,
    ):
        super().__init__()
        self.upscale = upscale

        # ── Stem ──────────────────────────────────────────────────────────
        self.stem = nn.Conv2d(in_ch, width, 3, padding=1, bias=True)

        # ── Encoder ───────────────────────────────────────────────────────
        # enc_blocks[i] NAFBlocks → stride-2 Conv (doubles channels)
        self.encoder_layers = nn.ModuleList()
        self.downsamplers   = nn.ModuleList()
        ch = width
        for n_blks in enc_blocks:
            self.encoder_layers.append(
                nn.Sequential(*[NAFBlock(ch) for _ in range(n_blks)])
            )
            self.downsamplers.append(
                nn.Conv2d(ch, ch * 2, 2, stride=2, bias=True)
            )
            ch *= 2

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(*[NAFBlock(ch) for _ in range(4)])

        # ── Decoder ───────────────────────────────────────────────────────
        self.upsamplers   = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for n_blks in dec_blocks:
            self.upsamplers.append(
                nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=True)
            )
            ch //= 2
            # After skip concat channel count doubles → fuse back to ch
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(ch * 2, ch, 1, bias=True),       # fuse skip
                    *[NAFBlock(ch) for _ in range(n_blks)],
                )
            )

        # ── Upsampling head ───────────────────────────────────────────────
        # ch should be back to `width` after all decoder levels
        self.upsample_head = PixelShufflePack(ch, in_ch, scale=upscale)

        # ── Soft-Clip output boundary ─────────────────────────────────────
        self.softclip = SoftClip(alpha=softclip_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)

        # Encoder: collect skip connections
        skips = []
        for enc, down in zip(self.encoder_layers, self.downsamplers):
            x = enc(x)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder: progressively fuse skips
        for up, dec, skip in zip(self.upsamplers, self.decoder_layers, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)   # skip concat
            x = dec(x)

        # PixelShuffle ×2 (all compute done at LR resolution)
        x = self.upsample_head(x)

        # Smooth [0, 1] output boundary
        x = self.softclip(x)

        return x


# ─────────────────────────────────────────────────────────────────────────────
# Model factory with optional torch.compile
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, device: torch.device) -> nn.Module:
    """
    Instantiate NAFNet from a config dict and optionally torch.compile it.

    Args:
        cfg    : dict with keys from model + train sections of train.yaml
        device : target torch.device

    Returns:
        model (possibly compiled)
    """
    model = NAFNet(
        in_ch      = cfg.get("channels", 1),
        width      = cfg["model"]["width"],
        enc_blocks = cfg["model"]["enc_blocks"],
        dec_blocks = cfg["model"]["dec_blocks"],
        upscale    = cfg["model"]["upscale"],
    ).to(device)

    if cfg["train"].get("compile", False):
        # Inductor optimisations for H100 Tensor Cores
        import torch._inductor.config as inductor_cfg
        inductor_cfg.conv_1x1_as_mm                = True
        inductor_cfg.coordinate_descent_tuning     = True
        inductor_cfg.epilogue_fusion               = True

        mode = cfg["train"].get("compile_mode", "max-autotune-no-cudagraphs")
        model = torch.compile(model, mode=mode, fullgraph=True)
        print(f"[NAFNet] torch.compile enabled  (mode={mode})")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[NAFNet] Trainable parameters: {n_params / 1e6:.2f} M")
    return model
