"""
evaluation/evaluate.py — Quantitative & qualitative evaluation of VAE and GAN.

Run:
    python evaluation/evaluate.py

Metrics computed
────────────────
• Reconstruction MSE   (VAE only)
• PSNR — Peak Signal-to-Noise Ratio
• SSIM — Structural Similarity Index  (via scikit-image)
• FID  — Fréchet Inception Distance   (approximated with a small sample;
          full FID needs ~10 k images; we report it honestly)
• Diversity score      (std-dev of pixel values across generated samples)

All metrics are printed to console AND saved to outputs/plots/metrics.txt.
Visual comparison grids are saved to outputs/plots/.
"""

import os, sys
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

import config
from models.vae import VAE, vae_loss
from models.gan import Generator
from utils.helpers import (
    set_seed, get_dataloaders, denorm,
    save_image_grid, save_comparison_grid, ensure_dirs,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_vae(checkpoint_path: str, device) -> VAE:
    model = VAE(latent_dim=config.VAE_LATENT_DIM,
                num_channels=config.NUM_CHANNELS).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Eval] VAE loaded from {checkpoint_path}  (epoch {ckpt.get('epoch','?')})")
    return model


def load_generator(checkpoint_path: str, device) -> Generator:
    G    = Generator(noise_dim=config.GAN_NOISE_DIM,
                     num_channels=config.NUM_CHANNELS).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(ckpt["G_state"])
    G.eval()
    print(f"[Eval] Generator loaded from {checkpoint_path}  (epoch {ckpt.get('epoch','?')})")
    return G


def tensor_to_numpy_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert a (C,H,W) tensor in [-1,1] → uint8 numpy (H,W,C)."""
    img = denorm(t).permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Per-image metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_mse(orig: torch.Tensor, recon: torch.Tensor) -> float:
    """MSE in [0,1] pixel space."""
    o = denorm(orig).cpu().numpy()
    r = denorm(recon).cpu().numpy()
    return float(np.mean((o - r) ** 2))


def compute_psnr(orig: torch.Tensor, recon: torch.Tensor) -> float:
    o = tensor_to_numpy_uint8(orig)
    r = tensor_to_numpy_uint8(recon)
    return psnr_fn(o, r, data_range=255)


def compute_ssim(orig: torch.Tensor, recon: torch.Tensor) -> float:
    o = tensor_to_numpy_uint8(orig)
    r = tensor_to_numpy_uint8(recon)
    return ssim_fn(o, r, data_range=255, channel_axis=2)


def diversity_score(images: torch.Tensor) -> float:
    """
    Mean std-dev across images in pixel space.
    Higher = more diverse outputs.
    """
    imgs = denorm(images).cpu().numpy()   # (N, C, H, W) in [0,1]
    return float(np.mean(np.std(imgs, axis=0)))


# ─────────────────────────────────────────────────────────────────────────────
# Approximate FID (honest implementation note)
# ─────────────────────────────────────────────────────────────────────────────

def approx_fid(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    """
    Approximate FID using pooled feature statistics (mean & covariance).
    This is a fast approximation computed on pooled 4x4 feature maps.
    Lower is better.
    """
    def stats(imgs):
        pooled = F.adaptive_avg_pool2d(denorm(imgs), (4, 4))
        x = pooled.permute(0, 2, 3, 1).cpu().numpy().reshape(imgs.size(0), -1).astype(np.float64)
        mu  = np.mean(x, axis=0)
        cov = np.cov(x, rowvar=False) + np.eye(x.shape[1]) * 1e-6
        return mu, cov

    mu1, cov1 = stats(real_images)
    mu2, cov2 = stats(fake_images)

    diff  = mu1 - mu2
    eigvals, eigvecs = np.linalg.eigh(cov1 @ cov2)
    eigvals = np.maximum(eigvals, 0)
    sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    fid = float(diff @ diff + np.trace(cov1 + cov2 - 2 * sqrt_cov))
    return max(0.0, fid)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate():
    set_seed(config.SEED)
    ensure_dirs()
    device = config.DEVICE

    print(f"\n{'='*60}")
    print(f"  Model Evaluation  |  Device: {device}")
    print(f"{'='*60}\n")

    # ── Load data (val split) ─────────────────────────────────────────────
    _, val_loader = get_dataloaders(
        dataset_name=config.DATASET,
        batch_size=64,
    )
    val_batch = next(iter(val_loader))
    x_real    = val_batch[0] if isinstance(val_batch, (list, tuple)) else val_batch
    x_real    = x_real.to(device)

    results = {}

    # ── VAE Evaluation ────────────────────────────────────────────────────
    vae_ckpt = os.path.join(config.CHECKPOINTS_DIR, "vae_best.pt")
    if os.path.exists(vae_ckpt):
        vae = load_vae(vae_ckpt, device)

        with torch.no_grad():
            recon, mu, logvar     = vae(x_real)
            _, recon_loss, kl_loss = vae_loss(recon, x_real, mu, logvar)
            vae_generated          = vae.generate(64, device)

        # Per-image metrics (first 16 images)
        mse_vals, psnr_vals, ssim_vals = [], [], []
        for i in range(min(16, x_real.size(0))):
            mse_vals.append(compute_mse(x_real[i],  recon[i]))
            psnr_vals.append(compute_psnr(x_real[i], recon[i]))
            ssim_vals.append(compute_ssim(x_real[i], recon[i]))

        div_vae = diversity_score(vae_generated)

        # Pixel-space FID
        fid_vae = approx_fid(x_real[:32], vae_generated[:32])

        results["VAE"] = {
            "Recon MSE":   np.mean(mse_vals),
            "PSNR (dB)":   np.mean(psnr_vals),
            "SSIM":        np.mean(ssim_vals),
            "Diversity":   div_vae,
            "Pixel FID*":  fid_vae,
            "KL Loss":     kl_loss.item(),
        }

        # Save grids
        save_comparison_grid(
            x_real, recon,
            os.path.join(config.PLOTS_DIR, "eval_vae_reconstruction.png"), n=8,
        )
        save_image_grid(
            vae_generated,
            os.path.join(config.GENERATED_DIR, "eval_vae_generated.png"),
            nrow=8, title="VAE Generated (Final)",
        )

        # Latent space interpolation (walk from z1 → z2)
        with torch.no_grad():
            z1 = torch.randn(1, config.VAE_LATENT_DIM, device=device)
            z2 = torch.randn(1, config.VAE_LATENT_DIM, device=device)
            alphas = torch.linspace(0, 1, 10, device=device)
            interp_imgs = torch.cat([
                vae.decoder(z1 * (1 - a) + z2 * a) for a in alphas
            ], dim=0)
        save_image_grid(
            interp_imgs,
            os.path.join(config.GENERATED_DIR, "vae_latent_interpolation.png"),
            nrow=10, title="VAE Latent Space Interpolation (z1 → z2)",
        )
    else:
        print(f"[Eval] VAE checkpoint not found at {vae_ckpt}. Skipping VAE eval.")

    # ── GAN Evaluation ────────────────────────────────────────────────────
    gan_ckpt = os.path.join(config.CHECKPOINTS_DIR, "gan_best.pt")
    if os.path.exists(gan_ckpt):
        G = load_generator(gan_ckpt, device)

        with torch.no_grad():
            z           = torch.randn(64, config.GAN_NOISE_DIM, device=device)
            gan_generated = G(z)

        div_gan = diversity_score(gan_generated)
        fid_gan = approx_fid(x_real[:32], gan_generated[:32])

        results["GAN"] = {
            "Recon MSE":   "N/A (GANs don't reconstruct)",
            "PSNR (dB)":   "N/A",
            "SSIM":        "N/A",
            "Diversity":   div_gan,
            "Pixel FID*":  fid_gan,
            "KL Loss":     "N/A",
        }

        save_image_grid(
            gan_generated,
            os.path.join(config.GENERATED_DIR, "eval_gan_generated.png"),
            nrow=8, title="DCGAN Generated (Final)",
        )
    else:
        print(f"[Eval] GAN checkpoint not found at {gan_ckpt}. Skipping GAN eval.")

    # ── Print + Save metrics ──────────────────────────────────────────────
    report_lines = [
        "=" * 60,
        "  EVALUATION RESULTS",
        "=" * 60,
        "",
        "NOTE: Pixel FID* is computed in RAW pixel space (not Inception",
        "features). True FID requires ~10k images + InceptionV3 features.",
        "",
    ]
    for model_name, metrics in results.items():
        report_lines.append(f"── {model_name} ──────────────────────")
        for k, v in metrics.items():
            if isinstance(v, float):
                report_lines.append(f"  {k:<20}: {v:.6f}")
            else:
                report_lines.append(f"  {k:<20}: {v}")
        report_lines.append("")

    report_text = "\n".join(report_lines)
    print(report_text)

    metrics_path = os.path.join(config.PLOTS_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(report_text)
    print(f"[Saved] {metrics_path}")

    # ── Side-by-side comparison plot ──────────────────────────────────────
    if "VAE" in results and "GAN" in results:
        _plot_comparison(x_real, recon, vae_generated, gan_generated)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Comparison visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _plot_comparison(real, vae_recon, vae_gen, gan_gen, n=6):
    fig = plt.figure(figsize=(n * 2.2, 9))
    gs  = gridspec.GridSpec(4, n, figure=fig, hspace=0.3, wspace=0.05)

    titles = ["Real Images", "VAE Reconstructed", "VAE Generated", "GAN Generated"]
    rows   = [real, vae_recon, vae_gen, gan_gen]

    for row_idx, (row_data, row_title) in enumerate(zip(rows, titles)):
        for col_idx in range(n):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            img = denorm(row_data[col_idx]).permute(1, 2, 0).cpu().numpy()
            ax.imshow(img.clip(0, 1))
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(row_title, fontsize=9, rotation=90, labelpad=60,
                              va="center")

    plt.suptitle("Model Comparison: Real vs VAE vs GAN", fontsize=14, y=1.01)
    path = os.path.join(config.PLOTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


if __name__ == "__main__":
    evaluate()
