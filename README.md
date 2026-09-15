# Generative Image Synthesis: VAE vs DCGAN

Implementation and comparative analysis of **Variational Autoencoders (VAE)** and **Deep Convolutional Generative Adversarial Networks (DCGAN)** in PyTorch for synthetic human face generation at 64x64 resolution.

---

## Visual Results: Real vs VAE vs DCGAN

Side-by-side visual comparison from the evaluation pipeline:

![Model Comparison Grid](outputs/plots/model_comparison.png)

*Figure 1: Row 1 — Real input faces; Row 2 — VAE reconstructions; Row 3 — VAE generated samples ($z \sim \mathcal{N}(0, \mathbf{I})$); Row 4 — DCGAN generated samples ($z \sim \mathcal{N}(0, \mathbf{I})$).*

---

## Table of Contents

- [Project Overview](#project-overview)
- [Experimental Results and Metrics](#experimental-results-and-metrics)
- [Architecture Details](#architecture-details)
- [VAE Results: Reconstruction and Latent Space Walk](#vae-results-reconstruction-and-latent-space-walk)
- [DCGAN Results: Adversarial Synthesis](#dcgan-results-adversarial-synthesis)
- [Training Loss Dynamics](#training-loss-dynamics)
- [Model Comparison](#model-comparison)
- [Analysis: Why VAE and GAN Yield Different Outputs](#analysis-why-vae-and-gan-yield-different-outputs)
- [Streamlit Web Interface](#streamlit-web-interface)
- [Project Directory Layout](#project-directory-layout)
- [Running the Project](#running-the-project)
- [Viva Voce Technical Discussion](#viva-voce-technical-discussion)
- [Summary Presentation Script](#summary-presentation-script)
- [References](#references)

---

## Project Overview

This project implements and analyzes two foundational generative deep-learning architectures:

1. **Variational Autoencoder (VAE)**: An explicit probabilistic latent-variable model that maximizes the Evidence Lower Bound (ELBO) using the **reparameterization trick** to encode images into a continuous 128-dimensional Gaussian latent space.
2. **Deep Convolutional GAN (DCGAN)**: An adversarial framework that trains a convolutional Generator against a Discriminator via a zero-sum minimax objective using **non-saturating loss** and **one-sided label smoothing**.

---

## Experimental Results and Metrics

The quantitative evaluation was conducted on held-out validation samples:

| Metric | Variational Autoencoder (VAE) | Deep Convolutional GAN (DCGAN) | Description |
|---|---|---|---|
| **Reconstruction MSE** | **0.018793** | N/A | Mean squared error in pixel space between original and reconstruction. |
| **PSNR** | **17.488 dB** | N/A | Peak Signal-to-Noise Ratio measuring reconstruction quality. |
| **SSIM** | **0.5461** | N/A | Structural Similarity Index evaluating luminance, contrast, and structure. |
| **Pixel Diversity** | **0.041822** | **0.042742** | Pixel variance across generated samples confirming good distribution coverage. |
| **Pixel FID** | **0.000000** | **0.997609** | Feature-pooled statistical distance between real and synthetic images. |
| **Final KL Loss** | **0.013979** | N/A | Relative entropy showing successful Gaussian latent space regularization. |
| **Visual Texture** | Smooth, blurred fine details | Sharp, high-frequency details | Consequence of MSE averaging vs. adversarial discriminator feedback. |
| **Latent Space** | Continuous, smooth interpolation | Unstructured latent mapping | VAE supports continuous walks between identities. |

---

## Architecture Details

### Variational Autoencoder (`models/vae.py`)

```
Input Image (3 x 64 x 64)
       │
       ▼ Conv2D (3 -> 64, kernel=4, stride=2, pad=1) + BatchNorm + LeakyReLU(0.2)   [64 x 32 x 32]
       ▼ Conv2D (64 -> 128, kernel=4, stride=2, pad=1) + BatchNorm + LeakyReLU(0.2) [128 x 16 x 16]
       ▼ Conv2D (128 -> 256, kernel=4, stride=2, pad=1) + BatchNorm + LeakyReLU(0.2)[256 x 8 x 8]
       ▼ Conv2D (256 -> 512, kernel=4, stride=2, pad=1) + BatchNorm + LeakyReLU(0.2)[512 x 4 x 4]
       ▼ Conv2D (512 -> 512, kernel=4, stride=1, pad=0) + LeakyReLU(0.2)            [512 x 1 x 1]
       │
       ├─────────────────────────────────┬─────────────────────────────────┐
       ▼ Linear (512 -> 128)             ▼ Linear (512 -> 128)             │
    Mean mu (128-d)                 Log-Variance log_sigma2 (128-d)        │
       └────────────────┬────────────────┘                                 │
                        ▼                                                  │
             Reparameterization Trick:                                     │
             z = mu + eps * exp(0.5 * log_sigma2),  eps ~ N(0, I)          │
                        │                                                  │
                        ▼ Latent Vector z (128-d)                          │
                        │                                                  │
       ▼ Linear (128 -> 512) + Reshape                                     │
       ▼ ConvTranspose2D (512 -> 256, kernel=4, stride=1, pad=0) + BatchNorm + ReLU  [256 x 4 x 4]
       ▼ ConvTranspose2D (256 -> 128, kernel=4, stride=2, pad=1) + BatchNorm + ReLU  [128 x 8 x 8]
       ▼ ConvTranspose2D (128 -> 64, kernel=4, stride=2, pad=1) + BatchNorm + ReLU   [64 x 16 x 16]
       ▼ ConvTranspose2D (64 -> 32, kernel=4, stride=2, pad=1) + BatchNorm + ReLU    [32 x 32 x 32]
       ▼ ConvTranspose2D (32 -> 3, kernel=4, stride=2, pad=1) + Tanh                 [3 x 64 x 64]
                        │
                        ▼
       Reconstructed Output Image x_hat in [-1, 1] (3 x 64 x 64)
```

**VAE Loss Formulation:**
$$\mathcal{L}_{\text{ELBO}} = \text{MSE}(x, \hat{x}) + \beta \cdot D_{\text{KL}}(q_\phi(z|x) \,\|\, p(z))$$

$$\mathcal{L}_{\text{ELBO}} = \frac{1}{N}\sum \|x - \hat{x}\|^2 - \frac{\beta}{2} \sum_{j=1}^{d} \left( 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)$$

---

### Deep Convolutional GAN (`models/gan.py`)

```
               GENERATOR                                      DISCRIMINATOR
    Noise z ~ N(0, I) (100-d)                          Input Image (3 x 64 x 64)
                │                                                  │
    ▼ ConvTranspose2D (100 -> 512) + BN + ReLU          ▼ Conv2D (3 -> 64, s=2) + LeakyReLU(0.2)
      [512 x 4 x 4]                                       [64 x 32 x 32] (No BN)
                │                                                  │
    ▼ ConvTranspose2D (512 -> 256) + BN + ReLU          ▼ Conv2D (64 -> 128, s=2) + BN + LeakyReLU
      [256 x 8 x 8]                                       [128 x 16 x 16]
                │                                                  │
    ▼ ConvTranspose2D (256 -> 128) + BN + ReLU          ▼ Conv2D (128 -> 256, s=2) + BN + LeakyReLU
      [128 x 16 x 16]                                     [256 x 8 x 8]
                │                                                  │
    ▼ ConvTranspose2D (128 -> 64) + BN + ReLU           ▼ Conv2D (256 -> 512, s=2) + BN + LeakyReLU
      [64 x 32 x 32]                                      [512 x 4 x 4]
                │                                                  │
    ▼ ConvTranspose2D (64 -> 3) + Tanh                  ▼ Conv2D (512 -> 1, s=1, pad=0) + Sigmoid
      [3 x 64 x 64]                                       [1 x 1 x 1]
                │                                                  │
                └───────────────► Synthetic Face ──────────────────┘
                                         │
                                         ▼
                             Real / Fake Probability in [0, 1]
```

**Adversarial Objectives:**
- **Discriminator Loss:**
  $$\mathcal{L}_D = -\mathbb{E}_{x} [\log D(x)] - \mathbb{E}_{z} [\log (1 - D(G(z)))]$$
  *(With real label set to 0.9 for one-sided label smoothing)*
- **Generator Loss (Non-saturating):**
  $$\mathcal{L}_G = -\mathbb{E}_{z} [\log D(G(z))]$$

---

## VAE Results: Reconstruction and Latent Space Walk

### Reconstruction Fidelity
Original input faces vs. VAE reconstructed faces:

![VAE Reconstruction Comparison](outputs/plots/eval_vae_reconstruction.png)

*Figure 2: Top row = Original ground-truth images; Bottom row = VAE reconstructions (MSE: 0.0188, PSNR: 17.49 dB).*

### Latent Space Manifold Walk (Interpolation)
Linear interpolation between two random points $z_1, z_2 \sim \mathcal{N}(0, \mathbf{I})$ demonstrates smooth topological continuity in the learned latent space:

![VAE Latent Interpolation](outputs/generated/vae_latent_interpolation.png)

*Figure 3: 10-step linear interpolation path in latent space between two face vectors.*

### VAE Generated Samples
Samples drawn directly from the Gaussian prior $z \sim \mathcal{N}(0, \mathbf{I})$:

![VAE Generated Grid](outputs/generated/eval_vae_generated.png)

*Figure 4: 64 synthetic faces generated by sampling the VAE prior.*

---

## DCGAN Results: Adversarial Synthesis

Samples produced by the DCGAN Generator from Gaussian noise vectors:

![GAN Generated Grid](outputs/generated/eval_gan_generated.png)

*Figure 5: 64 synthetic faces generated by DCGAN displaying defined facial features.*

---

## Training Loss Dynamics

### VAE Training Losses
The total ELBO loss decreases smoothly and stabilizes:

| VAE Total Loss | VAE Component Losses (Recon vs KL) |
|---|---|
| ![VAE Total Loss](outputs/plots/vae_total_loss.png) | ![VAE Component Losses](outputs/plots/vae_component_losses.png) |

### DCGAN Training Losses and Discriminator Scores
Loss interaction and discriminator predictions during training:

| DCGAN Generator and Discriminator Loss | Discriminator Real vs Fake Scores |
|---|---|
| ![GAN Losses](outputs/plots/gan_losses.png) | ![GAN Discriminator Scores](outputs/plots/gan_discriminator_scores.png) |

---

## Model Comparison

| Dimension | Variational Autoencoder (VAE) | Deep Convolutional GAN (DCGAN) |
|---|---|---|
| **Principle** | Explicit density approximation (ELBO) | Implicit density game (adversarial minimax) |
| **Loss** | Reconstruction MSE + KL Divergence | Binary Cross-Entropy (adversarial) |
| **Output Quality** | Smooth, slightly blurry fine features | Crisp, well-defined high-frequency details |
| **Diversity** | High (covers whole distribution support) | Moderate (susceptible to mode collapse) |
| **Image Reconstruction** | Direct via Encoder-Decoder | None without extra inference networks |
| **Latent Space** | Structured, Gaussian, interpolatable | Unstructured noise vector |
| **Training Stability** | High (standard gradient descent) | Requires balancing generator and discriminator |
| **Inference Time** | Fast single forward pass | Fast single forward pass |

---

## Analysis: Why VAE and GAN Yield Different Outputs

1. **Mean Squared Error Optimization (VAE):**
   The VAE decoder minimizes pixel-wise $\ell_2$ distance against training images. Mathematically, the expected value minimizing MSE under uncertainty is the **conditional mean** of the distribution. When multiple plausible textures exist for hair or skin, averaging them produces a smooth, blurry image.
2. **Adversarial Perceptual Feedback (GAN):**
   The GAN generator has no fixed pixel target. Instead, it must fool a discriminator that checks whether the image matches the distribution of real photographs. A blurry image is easily flagged as synthetic, forcing the generator to output sharp, high-frequency boundaries.

---

## Streamlit Web Interface

An interactive interface is provided in `app.py`:

```bash
streamlit run app.py
```

Features included:
- **VAE Generation**: Interactive face generation from latent space.
- **DCGAN Generation**: Synthetic face generation from noise.
- **VAE Reconstruction**: Ground truth vs. reconstructed face comparison.
- **Latent Interpolation**: Slider to walk continuously through the 128-d latent space.
- **Model Comparison**: Metric readouts and comparative plots.

---

## Project Directory Layout

```
VAE-vs-GAN/
├── config.py                 # Configuration and hyperparameters
├── requirements.txt          # Dependencies
├── run_all.py                # End-to-end execution script
├── app.py                    # Streamlit interface
├── README.md                 # Project documentation
├── report.md                 # Academic report and viva questions
├── .gitignore                # Excludes large weight files (>100MB)
├── data/
│   └── generate_face_dataset.py # Face dataset generator
├── models/
│   ├── vae.py                # VAE architecture and loss
│   └── gan.py                # DCGAN Generator and Discriminator
├── training/
│   ├── train_vae.py          # VAE training script
│   └── train_gan.py          # DCGAN training script
├── evaluation/
│   └── evaluate.py           # Metric calculation and plotting
└── outputs/
    ├── generated/            # Generated sample grids
    ├── reconstructions/      # Reconstruction comparisons
    └── plots/                # Loss curves and metrics.txt
```

---

## Running the Project

### 1. Setup Environment
```bash
git clone https://github.com/aditya29625/VAE-vs-GAN.git
cd VAE-vs-GAN
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Training and Evaluation
```bash
python run_all.py
```

### 3. Launch the Interface
```bash
streamlit run app.py
```

---

## Viva Voce Technical Discussion

### Q1: What is the reparameterization trick and why is it needed?
> **Answer:** In a VAE, the encoder outputs distribution parameters $\mu$ and $\log \sigma^2$. Sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ directly is a non-differentiable stochastic step that breaks backpropagation. The reparameterization trick expresses the sample as $z = \mu + \varepsilon \cdot \sigma$, where $\varepsilon \sim \mathcal{N}(0, \mathbf{I})$. This isolates the stochasticity in $\varepsilon$, making $z$ deterministic with respect to $\mu$ and $\sigma$ so gradients can flow normally.

### Q2: Why is the KL divergence term necessary in VAE?
> **Answer:** Without the KL divergence penalty, the encoder could map training points to isolated coordinates with $\sigma \to 0$, leaving unmapped gaps in latent space. Sampling from those gaps at inference time would produce invalid outputs. The KL term enforces a smooth Gaussian prior $\mathcal{N}(0, \mathbf{I})$ across the entire latent space, ensuring any sampled vector decodes to a valid image.

### Q3: Why are VAE images blurrier than GAN images?
> **Answer:** VAEs minimize an $\ell_2$ pixel reconstruction loss (MSE). When the network cannot be certain of the exact high-frequency detail, the mathematical minimum of MSE is the expected mean over all plausible details. The average of sharp variations is naturally blurred. GANs instead rely on a discriminator that rejects blurred images as fake, driving the generator to synthesize crisp details.

### Q4: What is mode collapse in GANs and how can it be identified?
> **Answer:** Mode collapse occurs when the generator produces only a limited variety of images that successfully fool the discriminator, ignoring the rest of the target distribution. It is detected when the diversity score (pixel variance across generated samples) drops significantly and different noise inputs produce nearly identical images.

### Q5: What is the non-saturating generator loss?
> **Answer:** The original minimax loss $\min_G \mathbb{E}[\log(1 - D(G(z)))]$ leads to vanishing gradients early in training when the discriminator easily detects fake images ($D(G(z)) \approx 0$). The non-saturating formulation maximizes $\log D(G(z))$, which provides large gradient signals early in training when the generator is performing poorly.

### Q6: Why use label smoothing for the discriminator?
> **Answer:** Setting the target for real images to 0.9 rather than 1.0 prevents the discriminator from becoming overconfident. Overconfidence causes vanishing gradients for the generator and destabilizes the adversarial training dynamics.

### Q7: Can a DCGAN reconstruct an input image directly?
> **Answer:** No. A standard GAN only has a Generator mapping latent codes to images ($G: \mathcal{Z} \to \mathcal{X}$). It lacks an encoder network to map images back to latent codes. Image reconstruction requires either iterative latent space optimization or architectures with bidirectional inference (such as BiGAN).

### Q8: What does $\beta$-VAE do?
> **Answer:** $\beta$-VAE scales the KL divergence term by a hyperparameter $\beta > 1$: $\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}$. This stronger regularization constraint encourages statistical independence among latent dimensions, leading to disentangled representations at the expense of slightly higher reconstruction error.

### Q9: Why are strided convolutions preferred over pooling in DCGAN?
> **Answer:** Pooling operations like MaxPool discard spatial location information deterministically. Strided convolutions and transposed convolutions allow the network to learn optimal spatial downsampling and upsampling filters end-to-end.

### Q10: How are generative models quantitatively evaluated?
> **Answer:** For reconstruction, **MSE, PSNR, and SSIM** are computed against ground truth. For unconditional synthesis, **Fréchet Inception Distance (FID)** measures the Wasserstein-2 distance between feature distributions of real and fake images, while **Inception Score (IS)** evaluates image clarity and class diversity.

---

## Summary Presentation Script

> *"Good morning. In this project, we implemented and evaluated two generative models from scratch in PyTorch: a Variational Autoencoder (VAE) and a Deep Convolutional GAN (DCGAN) for synthetic face synthesis at 64x64 resolution.*
>
> *For the VAE, we built a 5-layer convolutional encoder mapping into a 128-dimensional latent space $(\mu, \log \sigma^2)$. Using the reparameterization trick ($z = \mu + \varepsilon \cdot \sigma$), the sampling step remains differentiable. The decoder upsamples $z$ back to image space through 5 transposed convolutional layers. We trained it with the Evidence Lower Bound (ELBO), balancing pixel MSE with an analytical KL-divergence penalty that ensures a continuous Gaussian latent manifold.*
>
> *For the DCGAN, we implemented a zero-sum minimax game between a convolutional Generator and Discriminator. The Generator maps 100-dimensional noise into images, while the Discriminator classifies authenticity using strided convolutions. We stabilized training with non-saturating generator loss and one-sided label smoothing.*
>
> *Our results highlight the core trade-off between the two approaches: the VAE provides stable training, reliable coverage without mode collapse, and direct image reconstruction (achieving 0.0188 MSE and 17.49 dB PSNR) with smooth latent interpolation, but produces slightly blurred outputs due to $\ell_2$ averaging. The DCGAN produces significantly sharper facial features because its discriminator penalizes blur, though it lacks a direct reconstruction mechanism.*
>
> *We also created an interactive Streamlit application to demonstrate real-time face generation, reconstruction, and latent space traversal. Thank you."*

---

## References

1. Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *arXiv:1312.6114*.
2. Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. *arXiv:1511.06434*.
3. Goodfellow, I., et al. (2014). Generative Adversarial Networks. *NeurIPS*.
4. Higgins, I., et al. (2017). $\beta$-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. *ICLR*.
