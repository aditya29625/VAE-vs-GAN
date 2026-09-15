"""
config.py — Global configuration for the project.
All hyperparameters live here so every training script reads the same values.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import torch

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET        = "celeba"          # "celeba" | "cifar10"
DATA_DIR       = "./data"
IMAGE_SIZE     = 64                # resize to 64×64
NUM_CHANNELS   = 3
NUM_WORKERS    = 0

# ── VAE ───────────────────────────────────────────────────────────────────────
VAE_LATENT_DIM  = 128
VAE_BATCH_SIZE  = 64
VAE_EPOCHS      = 5
VAE_LR          = 1e-3
VAE_BETA        = 1.0              # weight on KL term (β-VAE: set >1 for disentanglement)

# ── GAN ───────────────────────────────────────────────────────────────────────
GAN_NOISE_DIM   = 100
GAN_BATCH_SIZE  = 64
GAN_EPOCHS      = 5
GAN_LR_G        = 2e-4
GAN_LR_D        = 2e-4
GAN_BETA1       = 0.5              # Adam β₁ for GANs (standard DCGAN value)
GAN_BETA2       = 0.999

# ── Feature-map depths ───────────────────────────────────────────────────────
GAN_NGF         = 64               # generator base feature-maps
GAN_NDF         = 64               # discriminator base feature-maps

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR            = "./outputs"
GENERATED_DIR         = "./outputs/generated"
RECONSTRUCTIONS_DIR   = "./outputs/reconstructions"
PLOTS_DIR             = "./outputs/plots"
CHECKPOINTS_DIR       = "./outputs/checkpoints"
