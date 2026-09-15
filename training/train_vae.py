"""
training/train_vae.py — Complete VAE training loop.

Run:
    python training/train_vae.py

What this script does
─────────────────────
1. Loads the dataset (CelebA or CIFAR-10 fallback)
2. Builds the VAE model
3. Trains for VAE_EPOCHS, tracking recon + KL losses
4. Saves checkpoints and sample grids every 5 epochs
5. Plots and saves the training/validation loss curves
"""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim
from tqdm import tqdm

import config
from models.vae import VAE, vae_loss
from utils.helpers import (
    set_seed, get_dataloaders, save_image_grid,
    save_comparison_grid, plot_losses, ensure_dirs,
)


# ─────────────────────────────────────────────────────────────────────────────
# Training function
# ─────────────────────────────────────────────────────────────────────────────

def train_vae():
    set_seed(config.SEED)
    ensure_dirs()
    device = config.DEVICE
    print(f"\n{'='*60}")
    print(f"  VAE Training  |  Device: {device}  |  Epochs: {config.VAE_EPOCHS}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        dataset_name=config.DATASET,
        batch_size=config.VAE_BATCH_SIZE,
    )

    # ── Model & Optimiser ────────────────────────────────────────────────────
    model = VAE(
        latent_dim=config.VAE_LATENT_DIM,
        num_channels=config.NUM_CHANNELS,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.VAE_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[VAE] Total parameters: {total_params:,}\n")

    # ── Fixed noise for consistent sample grids ──────────────────────────────
    fixed_z = torch.randn(64, config.VAE_LATENT_DIM, device=device)

    # ── History ──────────────────────────────────────────────────────────────
    history = {
        "train_total": [], "train_recon": [], "train_kl": [],
        "val_total":   [], "val_recon":   [], "val_kl":   [],
    }

    best_val_loss = float("inf")
    start_time    = time.time()

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(1, config.VAE_EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        t_total = t_recon = t_kl = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{config.VAE_EPOCHS} [Train]",
                    leave=False, ncols=90)
        for batch in pbar:
            # Handle both (images, labels) tuples and plain image tensors
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, rl, kl      = vae_loss(recon, x, mu, logvar, beta=config.VAE_BETA)
            loss.backward()
            # Gradient clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            t_total += loss.item()
            t_recon += rl.item()
            t_kl    += kl.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "recon": f"{rl.item():.4f}", "kl": f"{kl.item():.4f}"})

        t_total /= n_batches
        t_recon /= n_batches
        t_kl    /= n_batches

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        v_total = v_recon = v_kl = 0.0
        v_batches = 0
        val_originals = val_reconstructions = None

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device)

                recon, mu, logvar = model(x)
                loss, rl, kl      = vae_loss(recon, x, mu, logvar, beta=config.VAE_BETA)

                v_total += loss.item()
                v_recon += rl.item()
                v_kl    += kl.item()
                v_batches += 1

                # Keep first batch for visualisation
                if val_originals is None:
                    val_originals       = x.cpu()
                    val_reconstructions = recon.cpu()

        v_total /= v_batches
        v_recon /= v_batches
        v_kl    /= v_batches

        # ── Record history ────────────────────────────────────────────────
        history["train_total"].append(t_total)
        history["train_recon"].append(t_recon)
        history["train_kl"].append(t_kl)
        history["val_total"].append(v_total)
        history["val_recon"].append(v_recon)
        history["val_kl"].append(v_kl)

        elapsed = (time.time() - start_time) / 60
        print(f"Epoch {epoch:03d}/{config.VAE_EPOCHS}  "
              f"Train [total={t_total:.4f} recon={t_recon:.4f} kl={t_kl:.4f}]  "
              f"Val [total={v_total:.4f} recon={v_recon:.4f} kl={v_kl:.4f}]  "
              f"Elapsed: {elapsed:.1f}m")

        scheduler.step()

        # ── Save samples every 5 epochs ───────────────────────────────────
        if epoch % 5 == 0 or epoch == 1:
            # Generated from fixed latent vectors
            with torch.no_grad():
                model.eval()
                gen_imgs = model.decoder(fixed_z)
            save_image_grid(
                gen_imgs,
                os.path.join(config.GENERATED_DIR, f"vae_generated_epoch{epoch:03d}.png"),
                nrow=8,
                title=f"VAE Generated — Epoch {epoch}",
            )
            # Reconstruction comparison
            if val_originals is not None:
                save_comparison_grid(
                    val_originals,
                    val_reconstructions,
                    os.path.join(config.RECONSTRUCTIONS_DIR, f"vae_recon_epoch{epoch:03d}.png"),
                    n=8,
                )

        # ── Save best checkpoint ──────────────────────────────────────────
        if v_total < best_val_loss:
            best_val_loss = v_total
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": v_total,
                "history": history,
            }, os.path.join(config.CHECKPOINTS_DIR, "vae_best.pt"))
            print(f"  ✓ Best model saved (val_loss={v_total:.4f})")

    # ── Save final checkpoint ────────────────────────────────────────────────
    torch.save({
        "epoch": config.VAE_EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }, os.path.join(config.CHECKPOINTS_DIR, "vae_final.pt"))

    # ── Plot loss curves ─────────────────────────────────────────────────────
    plot_losses(
        {"Train Total": history["train_total"], "Val Total": history["val_total"]},
        os.path.join(config.PLOTS_DIR, "vae_total_loss.png"),
        title="VAE Total Loss (Recon + KL)",
    )
    plot_losses(
        {
            "Train Recon": history["train_recon"], "Val Recon": history["val_recon"],
            "Train KL":    history["train_kl"],    "Val KL":   history["val_kl"],
        },
        os.path.join(config.PLOTS_DIR, "vae_component_losses.png"),
        title="VAE Component Losses",
    )

    total_time = (time.time() - start_time) / 60
    print(f"\n[VAE] Training complete in {total_time:.1f} min")
    print(f"[VAE] Best val loss: {best_val_loss:.4f}")
    print(f"[VAE] Checkpoints → {config.CHECKPOINTS_DIR}")
    print(f"[VAE] Plots        → {config.PLOTS_DIR}")

    return model, history


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_vae()
