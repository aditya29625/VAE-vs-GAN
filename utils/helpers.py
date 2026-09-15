"""
utils/helpers.py — Shared utility functions used across the project.
"""

import os
import random
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset Loading
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(image_size: int = config.IMAGE_SIZE) -> transforms.Compose:
    """Standard image transforms: resize → crop → tensor → normalize to [-1,1]."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],   # mean per channel
                             [0.5, 0.5, 0.5]),   # std  per channel
    ])


def get_dataloaders(dataset_name: str = config.DATASET,
                    batch_size: int = config.VAE_BATCH_SIZE,
                    image_size: int = config.IMAGE_SIZE):
    """
    Returns (train_loader, val_loader).

    Priority order:
      1. CelebA  (requires manual download — see README)
      2. CIFAR-10 (auto-downloaded, 3-channel, used as fallback)
    """
    transform = get_transforms(image_size)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    faces_path = os.path.join(config.DATA_DIR, "faces")
    if os.path.isdir(faces_path):
        print("[Dataset] Using local faces dataset from", faces_path)
        full_dataset = datasets.ImageFolder(faces_path, transform=transform)
    elif dataset_name.lower() == "celeba":
        celeba_path = os.path.join(config.DATA_DIR, "celeba")
        if os.path.isdir(celeba_path):
            print("[Dataset] Using CelebA from", celeba_path)
            full_dataset = datasets.ImageFolder(celeba_path, transform=transform)
        else:
            print("[Dataset] CelebA not found locally. Attempting torchvision download…")
            try:
                full_dataset = datasets.CelebA(
                    root=config.DATA_DIR,
                    split="all",
                    download=True,
                    transform=transform,
                )
            except Exception as e:
                print(f"[Dataset] CelebA download failed ({e}).")
                print("[Dataset] Falling back to CIFAR-10.")
                dataset_name = "cifar10"

    if dataset_name.lower() == "cifar10":
        print("[Dataset] Using CIFAR-10 (3-channel, 32×32 → resized to", image_size, ")")
        full_dataset = datasets.CIFAR10(
            root=config.DATA_DIR,
            train=True,
            download=True,
            transform=transform,
        )

    # 90 / 10 train-val split
    n_train = int(0.9 * len(full_dataset))
    n_val   = len(full_dataset) - n_train
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.SEED)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    print(f"[Dataset] Train: {n_train} | Val: {n_val} | Batches/epoch: {len(train_loader)}")
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# 3. Image Visualization
# ─────────────────────────────────────────────────────────────────────────────

def denorm(tensor: torch.Tensor) -> torch.Tensor:
    """Undo the [-1,1] normalization → [0,1] for display."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def save_image_grid(tensor: torch.Tensor, path: str, nrow: int = 8, title: str = ""):
    """Save a batch of images as a grid to *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = torchvision.utils.make_grid(denorm(tensor.cpu()), nrow=nrow, padding=2)
    np_img = grid.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(np_img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def save_comparison_grid(originals: torch.Tensor,
                         reconstructions: torch.Tensor,
                         path: str,
                         n: int = 8):
    """Save side-by-side original vs. reconstruction grid."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    originals       = denorm(originals[:n].cpu())
    reconstructions = denorm(reconstructions[:n].cpu())

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 5))
    for i in range(n):
        axes[0, i].imshow(originals[i].permute(1, 2, 0).numpy())
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructions[i].permute(1, 2, 0).numpy())
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original",       fontsize=10, loc="left")
    axes[1, 0].set_title("Reconstructed",  fontsize=10, loc="left")
    plt.suptitle("VAE Reconstruction", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Loss Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_losses(losses: dict, path: str, title: str = "Training Loss"):
    """
    Plot one or more named loss curves.

    losses = {"Recon Loss": [...], "KL Loss": [...]}
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, vals in losses.items():
        ax.plot(vals, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


def ensure_dirs():
    """Create all output directories."""
    for d in [config.GENERATED_DIR,
              config.RECONSTRUCTIONS_DIR,
              config.PLOTS_DIR,
              config.CHECKPOINTS_DIR]:
        os.makedirs(d, exist_ok=True)
