"""
training/train_gan.py — Complete DCGAN training loop.

Run:
    python training/train_gan.py

What this script does
─────────────────────
1. Loads the dataset (CelebA or CIFAR-10 fallback)
2. Builds Generator + Discriminator
3. Alternating adversarial training loop:
   Step A — Update Discriminator: real batch → loss_real, fake batch → loss_fake
   Step B — Update Generator:     fool discriminator → loss_G
4. Saves checkpoints + sample grids every 5 epochs
5. Plots G/D loss curves
"""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim
from tqdm import tqdm

import config
from models.gan import Generator, Discriminator, generator_loss, discriminator_loss
from utils.helpers import (
    set_seed, get_dataloaders, save_image_grid,
    plot_losses, ensure_dirs,
)


def train_gan():
    set_seed(config.SEED)
    ensure_dirs()
    device = config.DEVICE
    print(f"\n{'='*60}")
    print(f"  DCGAN Training  |  Device: {device}  |  Epochs: {config.GAN_EPOCHS}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, _ = get_dataloaders(
        dataset_name=config.DATASET,
        batch_size=config.GAN_BATCH_SIZE,
    )

    # ── Models ───────────────────────────────────────────────────────────────
    G = Generator(
        noise_dim=config.GAN_NOISE_DIM,
        num_channels=config.NUM_CHANNELS,
        ngf=config.GAN_NGF,
    ).to(device)

    D = Discriminator(
        num_channels=config.NUM_CHANNELS,
        ndf=config.GAN_NDF,
    ).to(device)

    print(f"[GAN] Generator params:     {sum(p.numel() for p in G.parameters()):,}")
    print(f"[GAN] Discriminator params: {sum(p.numel() for p in D.parameters()):,}\n")

    # ── Optimisers ───────────────────────────────────────────────────────────
    opt_G = optim.Adam(G.parameters(), lr=config.GAN_LR_G,
                       betas=(config.GAN_BETA1, config.GAN_BETA2))
    opt_D = optim.Adam(D.parameters(), lr=config.GAN_LR_D,
                       betas=(config.GAN_BETA1, config.GAN_BETA2))

    # Fixed noise for consistent visualisation across epochs
    fixed_noise = torch.randn(64, config.GAN_NOISE_DIM, device=device)

    # ── History ──────────────────────────────────────────────────────────────
    history = {"G_loss": [], "D_loss": [], "D_real": [], "D_fake": []}

    best_g_loss  = float("inf")
    start_time   = time.time()

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(1, config.GAN_EPOCHS + 1):
        G.train(); D.train()

        epoch_g = epoch_d = epoch_dr = epoch_df = 0.0
        n_batches = 0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch:03d}/{config.GAN_EPOCHS}",
                    leave=False, ncols=100)

        for batch in pbar:
            x_real = batch[0] if isinstance(batch, (list, tuple)) else batch
            x_real = x_real.to(device)
            B      = x_real.size(0)

            # ── Step A: Train Discriminator ───────────────────────────────
            # (Train D more than G by doing this first with detached fakes)
            opt_D.zero_grad()

            # Real images
            real_pred = D(x_real)

            # Fake images (detach so gradients don't flow into G)
            z         = torch.randn(B, config.GAN_NOISE_DIM, device=device)
            x_fake    = G(z).detach()
            fake_pred = D(x_fake)

            loss_D = discriminator_loss(real_pred, fake_pred, device)
            loss_D.backward()
            opt_D.step()

            # ── Step B: Train Generator ───────────────────────────────────
            opt_G.zero_grad()

            z         = torch.randn(B, config.GAN_NOISE_DIM, device=device)
            x_fake    = G(z)
            fake_pred = D(x_fake)          # re-evaluate with updated D

            loss_G = generator_loss(fake_pred, device)
            loss_G.backward()
            opt_G.step()

            # Accumulate metrics
            epoch_g  += loss_G.item()
            epoch_d  += loss_D.item()
            epoch_dr += real_pred.mean().item()
            epoch_df += fake_pred.mean().item()
            n_batches += 1

            pbar.set_postfix({
                "G": f"{loss_G.item():.3f}",
                "D": f"{loss_D.item():.3f}",
                "D(x)": f"{real_pred.mean().item():.3f}",
                "D(G)": f"{fake_pred.mean().item():.3f}",
            })

        epoch_g  /= n_batches
        epoch_d  /= n_batches
        epoch_dr /= n_batches
        epoch_df /= n_batches

        history["G_loss"].append(epoch_g)
        history["D_loss"].append(epoch_d)
        history["D_real"].append(epoch_dr)
        history["D_fake"].append(epoch_df)

        elapsed = (time.time() - start_time) / 60
        print(f"Epoch {epoch:03d}/{config.GAN_EPOCHS}  "
              f"G={epoch_g:.4f}  D={epoch_d:.4f}  "
              f"D(real)={epoch_dr:.3f}  D(fake)={epoch_df:.3f}  "
              f"Elapsed: {elapsed:.1f}m")

        # ── Save sample grid every 5 epochs ──────────────────────────────
        if epoch % 5 == 0 or epoch == 1:
            G.eval()
            with torch.no_grad():
                fake_imgs = G(fixed_noise)
            save_image_grid(
                fake_imgs,
                os.path.join(config.GENERATED_DIR, f"gan_generated_epoch{epoch:03d}.png"),
                nrow=8,
                title=f"DCGAN Generated — Epoch {epoch}",
            )
            G.train()

        # ── Save best Generator checkpoint ────────────────────────────────
        if epoch_g < best_g_loss:
            best_g_loss = epoch_g
            torch.save({
                "epoch": epoch,
                "G_state": G.state_dict(),
                "D_state": D.state_dict(),
                "opt_G":   opt_G.state_dict(),
                "opt_D":   opt_D.state_dict(),
                "history": history,
            }, os.path.join(config.CHECKPOINTS_DIR, "gan_best.pt"))
            print(f"  ✓ Best GAN saved (G_loss={epoch_g:.4f})")

    # ── Final checkpoint ─────────────────────────────────────────────────────
    torch.save({
        "epoch":   config.GAN_EPOCHS,
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "history": history,
    }, os.path.join(config.CHECKPOINTS_DIR, "gan_final.pt"))

    # ── Loss curves ──────────────────────────────────────────────────────────
    plot_losses(
        {"Generator Loss": history["G_loss"], "Discriminator Loss": history["D_loss"]},
        os.path.join(config.PLOTS_DIR, "gan_losses.png"),
        title="DCGAN Training Losses",
    )
    plot_losses(
        {"D(real)": history["D_real"], "D(fake)": history["D_fake"]},
        os.path.join(config.PLOTS_DIR, "gan_discriminator_scores.png"),
        title="Discriminator Output Scores Over Training",
    )

    total_time = (time.time() - start_time) / 60
    print(f"\n[GAN] Training complete in {total_time:.1f} min")
    print(f"[GAN] Checkpoints → {config.CHECKPOINTS_DIR}")
    print(f"[GAN] Plots        → {config.PLOTS_DIR}")

    return G, D, history


if __name__ == "__main__":
    train_gan()
