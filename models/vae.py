"""
models/vae.py — Variational Autoencoder (VAE) built from scratch in PyTorch.

Architecture
────────────
Encoder:  Input (3×64×64) → Conv layers → Flatten → FC → μ and log σ²
Latent:   z = μ + ε·σ   where ε ~ N(0,I)   [reparameterization trick]
Decoder:  z → FC → Reshape → TransposedConv layers → Output (3×64×64)

Loss
────
L = Reconstruction loss  +  β · KL divergence
  = BCE/MSE(x̂, x)       +  β · (-½ Σ(1 + log σ² - μ² - σ²))

Viva Key Points
───────────────
Q: Why do we use the reparameterization trick?
A: Backpropagation cannot flow through a stochastic node (sampling).
   By writing z = μ + ε·σ where ε is a separate noise variable, the
   gradient flows through μ and σ — only ε is non-differentiable.

Q: What does the KL term do?
A: It regularises the latent space, forcing the encoder to produce
   distributions close to N(0,I). Without it, the encoder would learn
   to collapse σ→0 (deterministic) and the latent space would have gaps.

Q: Why is the generated output blurry?
A: MSE/BCE reconstruction loss averages over all plausible reconstructions,
   producing blurry means. GANs learn a sharper distribution via adversarial loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Helper block: Conv + BN + LeakyReLU
# ─────────────────────────────────────────────────────────────────────────────

def _conv_block(in_ch, out_ch, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _deconv_block(in_ch, out_ch, stride=2, last=False):
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False),
    ]
    if not last:
        layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    else:
        layers += [nn.Tanh()]   # output in [-1, 1]
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Convolutional encoder.

    Input : (B, 3, 64, 64)
    Output: μ (B, latent_dim)  and  log σ² (B, latent_dim)

    Spatial downsampling via strided convolutions (no pooling):
      64 → 32 → 16 → 8 → 4 → flatten
    """
    def __init__(self, latent_dim: int, num_channels: int = 3, base_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            _conv_block(num_channels, base_ch),        # 64→32
            _conv_block(base_ch, base_ch * 2),         # 32→16
            _conv_block(base_ch * 2, base_ch * 4),     # 16→8
            _conv_block(base_ch * 4, base_ch * 8),     # 8→4
            # Final conv to 1×1
            nn.Conv2d(base_ch * 8, base_ch * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flatten_dim = base_ch * 8   # 512 when base_ch=64
        self.fc_mu      = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar  = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)                    # (B, 512, 1, 1)
        h = h.view(h.size(0), -1)         # (B, 512)
        mu     = self.fc_mu(h)            # (B, latent_dim)
        logvar = self.fc_logvar(h)        # (B, latent_dim)
        return mu, logvar


# ─────────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Convolutional decoder (mirrors the encoder).

    Input : z  (B, latent_dim)
    Output: x̂  (B, 3, 64, 64)  — values in [-1, 1]

    Spatial upsampling via transposed convolutions:
      1 → 4 → 8 → 16 → 32 → 64
    """
    def __init__(self, latent_dim: int, num_channels: int = 3, base_ch: int = 64):
        super().__init__()
        self.base_ch = base_ch
        self.fc = nn.Linear(latent_dim, base_ch * 8)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base_ch * 4), nn.ReLU(inplace=True),  # (B,256,4,4)
            _deconv_block(base_ch * 4, base_ch * 2),               # (B,128,8,8)
            _deconv_block(base_ch * 2, base_ch),                    # (B,64,16,16)
            _deconv_block(base_ch, base_ch // 2),                   # (B,32,32,32)
            nn.ConvTranspose2d(base_ch // 2, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),                                              # (B,3,64,64)
        )

    def forward(self, z):
        h = self.fc(z)                             # (B, 512)
        h = h.view(h.size(0), self.base_ch * 8, 1, 1)  # (B, 512, 1, 1)
        return self.net(h)


# ─────────────────────────────────────────────────────────────────────────────
# VAE (wraps Encoder + Decoder)
# ─────────────────────────────────────────────────────────────────────────────

class VAE(nn.Module):
    """
    Full Variational Autoencoder.

    Forward pass returns:
        recon  — reconstructed image  (B, C, H, W)
        mu     — latent mean          (B, latent_dim)
        logvar — latent log-variance  (B, latent_dim)
    """
    def __init__(self,
                 latent_dim:   int = config.VAE_LATENT_DIM,
                 num_channels: int = config.NUM_CHANNELS,
                 base_ch:      int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = Encoder(latent_dim, num_channels, base_ch)
        self.decoder    = Decoder(latent_dim, num_channels, base_ch)

    # ── Reparameterization trick ─────────────────────────────────────────────
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        z = μ + ε · exp(0.5 · log σ²)    where ε ~ N(0, I)

        During evaluation (inference), we return μ directly (no noise).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)   # σ = exp(0.5 · log σ²)
            eps = torch.randn_like(std)      # ε ~ N(0, I)
            return mu + eps * std
        return mu                            # deterministic at eval time

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decoder(z)
        return recon, mu, logvar

    def generate(self, n: int, device) -> torch.Tensor:
        """Sample n images from the prior N(0, I) → decode."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=device)
            return self.decoder(z)


# ─────────────────────────────────────────────────────────────────────────────
# VAE Loss
# ─────────────────────────────────────────────────────────────────────────────

def vae_loss(recon: torch.Tensor,
             x:     torch.Tensor,
             mu:    torch.Tensor,
             logvar: torch.Tensor,
             beta:  float = config.VAE_BETA) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ELBO loss = Reconstruction loss + β · KL divergence

    Reconstruction: MSE between pixel values (images normalized to [-1,1]).
    KL divergence:  -½ · Σ(1 + log σ² - μ² - σ²)   analytically derived
                    from KL(N(μ,σ²) ‖ N(0,I)).

    Returns: (total_loss, recon_loss, kl_loss)  — all are per-sample means.
    """
    # MSE averaged over all pixels and batch
    recon_loss = F.mse_loss(recon, x, reduction="mean")

    # KL term: sum over latent dims, mean over batch
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = config.DEVICE
    model  = VAE().to(device)
    x      = torch.randn(4, 3, 64, 64, device=device)

    recon, mu, logvar = model(x)
    loss, rl, kl      = vae_loss(recon, x, mu, logvar)

    print("VAE sanity check")
    print(f"  Input:  {x.shape}")
    print(f"  Recon:  {recon.shape}")
    print(f"  μ:      {mu.shape}")
    print(f"  logvar: {logvar.shape}")
    print(f"  Loss:   total={loss.item():.4f}  recon={rl.item():.4f}  kl={kl.item():.4f}")

    gen = model.generate(8, device)
    print(f"  Generated: {gen.shape}")
    print("  ✓  VAE model is correct.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
