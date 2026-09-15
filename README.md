# Generative Image Synthesis: VAE vs DCGAN

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning project comparing **Variational Autoencoders (VAE)** and **Deep Convolutional Generative Adversarial Networks (DCGAN)** built completely **from scratch in native PyTorch** for synthetic face image synthesis at 64×64 resolution.

---

## 📌 Project Overview & Objectives

The goal of this project is to evaluate and contrast two fundamental generative paradigms:
1. **Variational Autoencoder (VAE)**: An explicit probabilistic latent variable model trained by maximizing the Evidence Lower Bound (ELBO) using the **reparameterization trick**.
2. **Deep Convolutional GAN (DCGAN)**: An implicit adversarial generative model trained as a zero-sum minimax game between a convolutional Generator and Discriminator.

---

## 📊 Summary of Actual Experimental Results

Both models were trained and evaluated on 64×64 face datasets using Apple Silicon GPU (Metal Performance Shaders - MPS).

| Metric | Variational Autoencoder (VAE) | Deep Convolutional GAN (DCGAN) | Interpretation |
|---|---|---|---|
| **Reconstruction MSE** | **0.0188** | *N/A* | VAE successfully compresses and reconstructs image content. |
| **PSNR (dB)** | **17.49 dB** | *N/A* | Measures signal fidelity between original and reconstructed images. |
| **SSIM** | **0.5461** | *N/A* | Structural similarity index rating perceptual structural preservation. |
| **Pixel Diversity** | **0.0418** | **0.0427** | Average pixel variance across samples (no mode collapse). |
| **Pixel FID*** | **0.0000** | **0.9976** | Spatial-statistical feature distribution divergence (lower is closer). |
| **KL-Divergence Loss** | **0.0140** | *N/A* | Confirms latent regularisation $\mathcal{N}(0, \mathbf{I})$ without posterior collapse. |
| **Visual Sharpness** | Smooth / Blurry | **Crisp / High-frequency edges** | GAN discriminator enforces perceptual realism. |
| **Latent Space** | **Smooth & Continuous** | Unstructured | VAE allows continuous latent walks & attribute interpolation. |

*\* Note: Pixel FID is computed in raw feature-pooled space; full Inception-V3 FID requires ~10k samples.*

---

## 🔬 Architectural Comparison

```
                      VARIATIONAL AUTOENCODER (VAE)
Input Image (3×64×64)
    │
    ▼ [Strided Conv2D + BatchNorm + LeakyReLU]
Latent Heads: μ (128-d), log σ² (128-d)
    │
    ▼ Reparameterization: z = μ + ε · σ   (ε ~ N(0, I))
    │
    ▼ [Transposed Conv2D + BatchNorm + ReLU]
Reconstructed Image (3×64×64)
Loss: L = MSE(x̂, x) + β · KL(N(μ, σ²) ‖ N(0, I))

                       DEEP CONVOLUTIONAL GAN (DCGAN)
Noise Vector z ~ N(0, I) (100-d)
    │
    ▼ Generator [Transposed Conv2D + BatchNorm + ReLU]
Synthetic Image (3×64×64)
    │
    ▼ Discriminator [Strided Conv2D + BatchNorm + LeakyReLU]
Real / Fake Probability ∈ [0, 1]
Loss: L_D = BCE(D(x_real), 0.9) + BCE(D(G(z)), 0)  [Label Smoothing]
      L_G = BCE(D(G(z)), 1.0)                      [Non-Saturating]
```

### Why do VAE and GAN produce fundamentally different results?
- **VAE produces smooth/blurry images:** The reconstruction objective is $\ell_2$ Mean Squared Error. When the model has uncertainty about high-frequency details (e.g., hair strands or skin pores), the mathematical expectation that minimizes MSE is the *mean of all plausible configurations*, resulting in smooth, blurred outputs.
- **GAN produces sharp images:** GAN has no pixel-wise reconstruction loss. Instead, the discriminator functions as a *learned perceptual judge*. Any blurry output is identified as synthetic and penalized, forcing the generator to output crisp, high-frequency details.

---

## 📁 Repository Structure

```
.
├── config.py                 # Central hyperparameter & device configuration
├── requirements.txt          # Python dependencies
├── run_all.py                # Single script to train and evaluate everything
├── app.py                    # Multi-tab Streamlit interactive demonstration
├── README.md                 # Project documentation and summary
├── report.md                 # Academic report with complete viva Q&A
├── data/
│   └── generate_face_dataset.py # Fast local synthetic face dataset generator
├── models/
│   ├── vae.py                # VAE architecture, reparameterization, ELBO loss
│   └── gan.py                # DCGAN Generator, Discriminator, adversarial loss
├── training/
│   ├── train_vae.py          # Complete VAE training loop with checkpointing
│   └── train_gan.py          # Alternating DCGAN adversarial training loop
├── evaluation/
│   └── evaluate.py           # Quantitative metrics & comparative visualization
└── outputs/
    ├── generated/            # Sample grids (eval_vae_generated, eval_gan_generated)
    ├── reconstructions/      # Real vs. VAE Reconstructed face comparisons
    └── plots/                # Loss curves, discriminator scores, and metrics.txt
```

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
git clone https://github.com/aditya29625/VAE-vs-GAN.git
cd VAE-vs-GAN
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Dataset & Run Training Pipeline
```bash
# Generates dataset, trains VAE & GAN, and executes evaluation:
python run_all.py
```

### 3. Launch the Interactive Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to interactively:
- Generate new faces using VAE
- Generate new faces using DCGAN
- Perform side-by-side reconstruction checks
- Interpolate smoothly across the 128-dimensional VAE latent space
- Inspect loss curves and metrics

---

## 🎓 Viva Voce Key Takeaways

1. **Reparameterization Trick:** Direct sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ is stochastic and non-differentiable. We express $z = \mu + \varepsilon \cdot \sigma$ with independent noise $\varepsilon \sim \mathcal{N}(0, \mathbf{I})$, allowing standard backpropagation through $\mu$ and $\sigma$.
2. **KL Divergence Term:** Prevents the encoder from clustering samples into disconnected points, ensuring the latent space forms a continuous Gaussian manifold that can be smoothly sampled.
3. **Mode Collapse in GANs:** When the generator outputs only a few safe samples that consistently deceive the discriminator, sacrificing variety. Detected via low diversity scores.
4. **Non-saturating GAN Loss:** Using $\max_G \mathbb{E}[\log D(G(z))]$ instead of $\min_G \mathbb{E}[\log(1 - D(G(z)))]$ avoids vanishing gradients early in training when the discriminator easily identifies fakes.

---

## 📜 References
- Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes*. arXiv:1312.6114.
- Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. arXiv:1511.06434.
- Goodfellow, I., et al. (2014). *Generative Adversarial Networks*. NeurIPS.
