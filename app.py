"""
app.py — Streamlit demo app for VAE + GAN face generation.

Run:
    streamlit run app.py

Features
────────
• Generate new faces with VAE (sample from prior)
• Generate new faces with GAN (sample from noise)
• Side-by-side model comparison
• VAE reconstruction viewer
• Latent space interpolation slider
"""

import os, sys
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

import config
from models.vae import VAE
from models.gan import Generator
from utils.helpers import denorm, get_dataloaders, set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VAE vs GAN — Face Generation",
    page_icon="🧠",
    layout="wide",
)

DEVICE = config.DEVICE


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached so it only runs once)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_vae_model():
    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, "vae_best.pt")
    if not os.path.exists(ckpt_path):
        return None
    model = VAE(latent_dim=config.VAE_LATENT_DIM,
                num_channels=config.NUM_CHANNELS).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@st.cache_resource
def load_gan_model():
    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, "gan_best.pt")
    if not os.path.exists(ckpt_path):
        return None
    G    = Generator(noise_dim=config.GAN_NOISE_DIM,
                     num_channels=config.NUM_CHANNELS).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    G.load_state_dict(ckpt["G_state"])
    G.eval()
    return G


@st.cache_resource
def load_val_batch():
    """Load one validation batch for the reconstruction demo."""
    try:
        _, val_loader = get_dataloaders(batch_size=16)
        batch = next(iter(val_loader))
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return x.to(DEVICE)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Single image tensor (C,H,W) in [-1,1] → PIL Image."""
    img = denorm(t).permute(1, 2, 0).cpu().numpy()
    img = (img.clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(img)


def show_grid(tensors, cols=8, caption=""):
    """Display a list of PIL images as a grid in Streamlit."""
    rows = [tensors[i:i+cols] for i in range(0, len(tensors), cols)]
    for row in rows:
        cols_st = st.columns(len(row))
        for col, img_t in zip(cols_st, row):
            col.image(tensor_to_pil(img_t), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# App layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("Generative Face Synthesis: VAE vs DCGAN")
st.markdown("""
A deep learning project evaluating **Variational Autoencoder (VAE)**
and **Deep Convolutional GAN (DCGAN)** for synthetic face generation.
""")

# Sidebar
st.sidebar.header("Controls")
n_generate = st.sidebar.slider("Images to generate", 4, 64, 16, step=4)
seed_val   = st.sidebar.number_input("Random seed", value=42, step=1)

vae_model = load_vae_model()
gan_model = load_gan_model()
val_batch = load_val_batch()

if vae_model is None and gan_model is None:
    st.warning("""
    **No trained models found.**

    Please train the models first:
    ```bash
    python training/train_vae.py
    python training/train_gan.py
    ```
    Then restart this app.
    """)
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "VAE Generation",
    "DCGAN Generation",
    "VAE Reconstruction",
    "Latent Interpolation",
    "Model Comparison",
])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — VAE Generation
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.header("VAE — Generate Faces from Random Latent Vectors")
    st.markdown("""
    **How it works:** We sample random vectors z ~ N(0, I) from the prior
    distribution and pass them through the trained decoder to generate new faces.

    VAE outputs tend to be **smooth but slightly blurry** because the MSE
    reconstruction loss penalises average pixel error, encouraging the model
    to predict the mean of all possible outputs.
    """)

    if vae_model is None:
        st.error("VAE checkpoint not found. Train the VAE first.")
    else:
        if st.button("Generate VAE Faces", key="vae_gen"):
            set_seed(int(seed_val))
            with torch.no_grad():
                z     = torch.randn(n_generate, config.VAE_LATENT_DIM, device=DEVICE)
                imgs  = vae_model.decoder(z)
            st.subheader(f"{n_generate} VAE-Generated Faces")
            show_grid(list(imgs), cols=8)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — GAN Generation
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.header("DCGAN — Generate Faces from Random Noise")
    st.markdown("""
    **How it works:** We sample random noise z ~ N(0, I) and pass it through
    the trained Generator network.

    GAN outputs tend to be **sharper and more realistic** because the
    adversarial loss forces the generator to fool a discriminator that has
    learned perceptual realism — not just pixel similarity.
    """)

    if gan_model is None:
        st.error("GAN checkpoint not found. Train the GAN first.")
    else:
        if st.button("Generate DCGAN Faces", key="gan_gen"):
            set_seed(int(seed_val))
            with torch.no_grad():
                z    = torch.randn(n_generate, config.GAN_NOISE_DIM, device=DEVICE)
                imgs = gan_model(z)
            st.subheader(f"{n_generate} DCGAN-Generated Faces")
            show_grid(list(imgs), cols=8)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — VAE Reconstruction
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.header("VAE — Image Reconstruction")
    st.markdown("""
    VAEs can **encode** real images into the latent space and **decode** them
    back. This demonstrates the compression + reconstruction capability.

    The reconstruction is never perfect — some detail is lost due to the
    bottleneck latent dimension and the KL regularisation.
    """)

    if vae_model is None:
        st.error("VAE checkpoint not found.")
    elif val_batch is None:
        st.error("Could not load validation data.")
    else:
        if st.button("Show Reconstructions", key="vae_recon"):
            n = min(8, val_batch.size(0))
            with torch.no_grad():
                recon, _, _ = vae_model(val_batch[:n])

            st.subheader("Original Images")
            show_grid(list(val_batch[:n]), cols=8)
            st.subheader("Reconstructed by VAE")
            show_grid(list(recon), cols=8)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Latent Interpolation
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.header("VAE — Latent Space Interpolation")
    st.markdown("""
    We linearly interpolate between **two random latent vectors** z₁ and z₂
    and decode each intermediate point. A smooth transition shows that the
    VAE has learned a **continuous, structured latent space**.
    """)

    if vae_model is None:
        st.error("VAE checkpoint not found.")
    else:
        steps = st.slider("Interpolation steps", 5, 20, 10)
        seed1 = st.number_input("Seed for z₁", value=1, step=1)
        seed2 = st.number_input("Seed for z₂", value=2, step=1)

        if st.button("Interpolate", key="interp"):
            with torch.no_grad():
                torch.manual_seed(int(seed1))
                z1 = torch.randn(1, config.VAE_LATENT_DIM, device=DEVICE)
                torch.manual_seed(int(seed2))
                z2 = torch.randn(1, config.VAE_LATENT_DIM, device=DEVICE)
                alphas = torch.linspace(0, 1, steps, device=DEVICE)
                interp = torch.cat([
                    vae_model.decoder(z1 * (1-a) + z2 * a) for a in alphas
                ], dim=0)

            st.subheader(f"Latent interpolation ({steps} steps): z₁ → z₂")
            show_grid(list(interp), cols=steps)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────

with tab5:
    st.header("VAE vs DCGAN — Model Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("VAE")
        st.markdown("""
        | Property | Detail |
        |---|---|
        | **Principle** | Probabilistic encoder-decoder |
        | **Loss** | MSE + KL divergence |
        | **Output quality** | Smooth, slightly blurry |
        | **Reconstruction** | Yes |
        | **Latent structure** | Continuous, interpolatable |
        | **Training** | Stable |
        | **Mode collapse** | No |
        | **Diversity** | High |
        | **Inference speed** | Fast |
        """)

    with col2:
        st.subheader("DCGAN")
        st.markdown("""
        | Property | Detail |
        |---|---|
        | **Principle** | Adversarial generator-discriminator |
        | **Loss** | Adversarial (BCE) |
        | **Output quality** | Sharp, realistic |
        | **Reconstruction** | No |
        | **Latent structure** | Unstructured |
        | **Training** | Sensitive to hyperparameters |
        | **Mode collapse** | Risk of partial collapse |
        | **Diversity** | Moderate |
        | **Inference speed** | Fast |
        """)

    st.markdown("---")
    st.subheader("Why do they produce different results?")
    st.markdown("""
    **VAE** optimises a pixel-level reconstruction loss (MSE). Since MSE measures
    the **average error** across all pixels, the model learns to predict the
    *mean* of all plausible outputs → blurry but diverse.

    **GAN** uses a learned adversarial loss. The discriminator acts as a
    *perceptual judge* — it rejects blurry fakes as "not real". This forces
    the generator to produce sharp, high-frequency details. However, the
    generator may exploit weaknesses in the discriminator, generating only
    a small variety of convincing images (mode collapse).

    **Key insight:** VAE maximises the Evidence Lower Bound (ELBO).
    GAN minimises a Jensen–Shannon divergence between real and fake distributions.
    """)

    # Show saved plots if they exist
    comparison_path = os.path.join(config.PLOTS_DIR, "model_comparison.png")
    if os.path.exists(comparison_path):
        st.image(comparison_path, caption="Model Comparison Grid", use_container_width=True)

    for plot_name, title in [
        ("vae_total_loss.png",          "VAE Training Loss"),
        ("gan_losses.png",              "GAN Training Losses"),
        ("gan_discriminator_scores.png","Discriminator Scores"),
    ]:
        p = os.path.join(config.PLOTS_DIR, plot_name)
        if os.path.exists(p):
            st.image(p, caption=title, use_container_width=True)

    metrics_path = os.path.join(config.PLOTS_DIR, "metrics.txt")
    if os.path.exists(metrics_path):
        st.subheader("Quantitative Metrics")
        with open(metrics_path) as f:
            st.code(f.read())

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Device:** `{DEVICE}`")
st.sidebar.markdown("**Dataset:** " + config.DATASET.upper())
st.sidebar.markdown("**VAE latent dim:** " + str(config.VAE_LATENT_DIM))
st.sidebar.markdown("**GAN noise dim:** " + str(config.GAN_NOISE_DIM))
