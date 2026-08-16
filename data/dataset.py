"""
data/dataset.py
───────────────
Dataset and DataLoader utilities for the speckle noise restoration pipeline.

Data format
-----------
  Training:
    dataset/train/train/GT/       – 3200 × (256, 256) float32 .npy  ∈ [0, 1]
    dataset/train/train/NoisyLR/  – 3200 × (128, 128) float32 .npy  (unclipped)

  Test (no GT):
    dataset/Test_NoisyLR/NoisyLR/ – 400  × (128, 128) float32 .npy

Key design choices
------------------
  • Raw float32 loaded with np.load; NO clipping applied to LR inputs.
  • mmap_mode='r' avoids loading the entire dataset into RAM.
  • pin_memory=True enables non-blocking DMA transfers over PCIe.
  • 90 / 10 deterministic train / val split from the 3 200 training pairs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ─────────────────────────────────────────────────────────────────────────────
# Training / Validation dataset
# ─────────────────────────────────────────────────────────────────────────────

class SpeckleDataset(Dataset):
    """
    Paired dataset of noisy low-resolution inputs and clean high-resolution
    ground-truth images stored as float32 NumPy arrays (.npy).

    Args:
        gt_dir   (str | Path): Directory containing GT .npy files.
        lr_dir   (str | Path): Directory containing NoisyLR .npy files.
        indices  (list[int] | None): Subset indices for train/val splitting.
                                     None → use all files.
    """

    def __init__(
        self,
        gt_dir: str | Path,
        lr_dir: str | Path,
        indices: Optional[list] = None,
    ):
        self.gt_dir = Path(gt_dir)
        self.lr_dir = Path(lr_dir)

        # Collect sorted file stems (e.g. "000000", "000001", …)
        all_stems = sorted(
            p.stem for p in self.gt_dir.glob("*.npy")
        )
        if indices is not None:
            all_stems = [all_stems[i] for i in indices]
        self.stems = all_stems

        if len(self.stems) == 0:
            raise RuntimeError(
                f"No .npy files found in GT dir: {self.gt_dir}"
            )

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]

        # Load without copying entire file into RAM (memory-mapped read)
        lr_arr = np.load(self.lr_dir / f"{stem}.npy", mmap_mode="r")
        gt_arr = np.load(self.gt_dir / f"{stem}.npy", mmap_mode="r")

        # (H, W) → (1, H, W) float32; LR stays unclipped
        lr = torch.from_numpy(np.array(lr_arr, dtype=np.float32)).unsqueeze(0)
        gt = torch.from_numpy(np.array(gt_arr, dtype=np.float32)).unsqueeze(0)

        return lr, gt


# ─────────────────────────────────────────────────────────────────────────────
# Test dataset (no ground truth)
# ─────────────────────────────────────────────────────────────────────────────

class SpeckleTestDataset(Dataset):
    """
    Test dataset: unclipped LR inputs only (no GT available).

    Args:
        lr_dir (str | Path): Directory containing NoisyLR .npy files.
    """

    def __init__(self, lr_dir: str | Path):
        self.lr_dir = Path(lr_dir)
        self.stems = sorted(p.stem for p in self.lr_dir.glob("*.npy"))

        if len(self.stems) == 0:
            raise RuntimeError(
                f"No .npy files found in test dir: {self.lr_dir}"
            )

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        stem = self.stems[idx]
        lr_arr = np.load(self.lr_dir / f"{stem}.npy", mmap_mode="r")
        lr = torch.from_numpy(np.array(lr_arr, dtype=np.float32)).unsqueeze(0)
        return lr, stem  # also return stem for output file naming


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    gt_dir: str,
    lr_dir: str,
    batch_size: int = 16,
    val_split: float = 0.1,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from a flat directory of .npy pairs.

    Applies a deterministic 90/10 split (seed=42) so val metrics are
    reproducible across runs.

    Returns:
        (train_loader, val_loader)
    """
    # Determine all indices then split deterministically
    all_stems = sorted(p.stem for p in Path(gt_dir).glob("*.npy"))
    n_total   = len(all_stems)
    n_val     = max(1, int(n_total * val_split))
    n_train   = n_total - n_val

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_total).tolist()
    train_idx = shuffled[:n_train]
    val_idx   = shuffled[n_train:]

    train_ds = SpeckleDataset(gt_dir, lr_dir, indices=train_idx)
    val_ds   = SpeckleDataset(gt_dir, lr_dir, indices=val_idx)

    loader_kwargs = dict(
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    return train_loader, val_loader


def build_test_loader(
    lr_dir: str,
    batch_size: int = 64,
    num_workers: int = 16,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for the unlabelled test set.

    Args:
        lr_dir     : Path to Test_NoisyLR/NoisyLR/
        batch_size : Micro-batch size for inference (default 64).
        num_workers: Parallel np.load threads.
    """
    test_ds = SpeckleTestDataset(lr_dir)
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )
