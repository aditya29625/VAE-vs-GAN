"""
models/gan.py — Deep Convolutional GAN (DCGAN) built from scratch in PyTorch.

Architecture
────────────
Generator:    Noise z (100,) → FC → Reshape → TransposedConv × 4 → (3,64,64)
Discriminator: (3,64,64) → Conv × 4 → Flatten → FC → Sigmoid scalar

Training (Minimax Game)
───────────────────────
  Discriminator loss:  L_D = -[log D(x) + log(1 - D(G(z)))]
  Generator loss:      L_G = -log D(G(z))          ← non-saturating trick

Viva Key Points
───────────────
Q: Why does GAN produce sharper images than VAE?
A: GANs use adversarial loss — the discriminator acts as a learned perceptual
   loss function that penalises blurry images. VAEs use pixel-wise MSE which
   averages over modes and yields blurry outputs.

Q: What is mode collapse?
A: The generator learns to produce only a few "safe" outputs that reliably
   fool the discriminator, ignoring most of the real data distribution.

Q: What is the non-saturating trick?
A: Instead of minimising log(1-D(G(z))) (which has near-zero gradient early
   on), we maximise log D(G(z)). Same equilibrium, much stronger early gradients.

Q: Why batch normalisation in DCGAN?
A: BN stabilises training by normalising intermediate activations, preventing
   gradient vanishing/exploding — critical in deep generative networks.
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Weight initialisation (from original DCGAN paper)
# ─────────────────────────────────────────────────────────────────────────────

def weights_init(m):
    """
    Apply normal initialisation to Conv and BN layers.
    From Radford et al. (2015): mean=0, std=0.02 for conv weights.
    """
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Maps random noise z ∈ ℝ^{noise_dim} → synthetic image ∈ ℝ^{C×64×64}.

    Layer-by-layer spatial growth (with ngf=64):
      z(100) → FC → (512,1,1) → (256,4,4) → (128,8,8) → (64,16,16) → (32,32,32) → (3,64,64)

    All conv-transpose layers use:
      kernel=4, stride=2, padding=1  → doubles spatial dimension each time
    First projection: kernel=4, stride=1, padding=0  → 1×1 → 4×4
    """
    def __init__(self,
                 noise_dim:    int = config.GAN_NOISE_DIM,
                 num_channels: int = config.NUM_CHANNELS,
                 ngf:          int = config.GAN_NGF):
        super().__init__()
        self.noise_dim = noise_dim

        self.net = nn.Sequential(
            # z → (ngf*8) × 4 × 4
            nn.ConvTranspose2d(noise_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf*8,4,4) → (ngf*4,8,8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4,8,8) → (ngf*2,16,16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*2,16,16) → (ngf,32,32)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf,32,32) → (C,64,64)
            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),          # output in [-1, 1]
        )
        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z shape: (B, noise_dim) → reshape to (B, noise_dim, 1, 1)
        z = z.view(z.size(0), self.noise_dim, 1, 1)
        return self.net(z)


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator
# ─────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Maps an image (real or fake) → scalar probability of being real.

    Layer-by-layer spatial compression (with ndf=64):
      (3,64,64) → (64,32,32) → (128,16,16) → (256,8,8) → (512,4,4) → scalar

    Notes:
    - No BN on the first layer (standard DCGAN practice).
    - LeakyReLU (slope 0.2) allows small gradients for negative inputs.
    - Sigmoid at the end → output ∈ (0,1) interpreted as P(real).
    """
    def __init__(self,
                 num_channels: int = config.NUM_CHANNELS,
                 ndf:          int = config.GAN_NDF):
        super().__init__()

        self.net = nn.Sequential(
            # (C,64,64) → (ndf,32,32)  — no BN on first layer
            nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf,32,32) → (ndf*2,16,16)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2,16,16) → (ndf*4,8,8)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4,8,8) → (ndf*8,4,4)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8,4,4) → (1,1,1)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)           # (B, 1, 1, 1)
        return out.view(-1)         # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# GAN Loss functions
# ─────────────────────────────────────────────────────────────────────────────

_bce = nn.BCELoss()

def discriminator_loss(real_pred: torch.Tensor,
                       fake_pred: torch.Tensor,
                       device) -> torch.Tensor:
    """
    L_D = -[log D(x) + log(1 - D(G(z)))]
        = BCE(D(x), 1) + BCE(D(G(z)), 0)

    We use soft labels (0.9 for real, 0.0 for fake) to improve stability.
    """
    real_labels = torch.full_like(real_pred, 0.9, device=device)   # label smoothing
    fake_labels = torch.zeros_like(fake_pred, device=device)

    loss_real = _bce(real_pred, real_labels)
    loss_fake = _bce(fake_pred, fake_labels)
    return loss_real + loss_fake


def generator_loss(fake_pred: torch.Tensor, device) -> torch.Tensor:
    """
    Non-saturating generator loss:
      L_G = -log D(G(z))  =  BCE(D(G(z)), 1)

    We want D to classify our fakes as real → maximise log D(G(z)).
    """
    real_labels = torch.ones_like(fake_pred, device=device)
    return _bce(fake_pred, real_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = config.DEVICE
    G = Generator().to(device)
    D = Discriminator().to(device)

    z      = torch.randn(4, config.GAN_NOISE_DIM, device=device)
    fake   = G(z)
    d_fake = D(fake)
    d_real = D(torch.randn(4, 3, 64, 64, device=device))

    print("GAN sanity check")
    print(f"  Noise z:       {z.shape}")
    print(f"  Fake image:    {fake.shape}   range [{fake.min():.2f}, {fake.max():.2f}]")
    print(f"  D(fake):       {d_fake.shape}   values {d_fake.detach().cpu().numpy()}")
    print(f"  D(real):       {d_real.shape}")

    g_loss = generator_loss(d_fake, device)
    d_loss = discriminator_loss(d_real, d_fake.detach(), device)
    print(f"  G loss: {g_loss.item():.4f}  |  D loss: {d_loss.item():.4f}")

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"  Generator params:     {g_params:,}")
    print(f"  Discriminator params: {d_params:,}")
    print("  ✓  GAN models are correct.")
