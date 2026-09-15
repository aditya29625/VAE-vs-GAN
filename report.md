# Academic Report: Generative Image Synthesis using VAE and GAN

**Course:** Deep Learning / Generative AI  
**Topic:** Synthetic Face Image Generation  
**Models:** Variational Autoencoder (VAE) + Deep Convolutional GAN (DCGAN)  
**Framework:** PyTorch (from scratch — no pretrained generative models)

---

## 1. Introduction

Generative models learn the underlying probability distribution of a dataset and can produce new samples that were not present in training data. This project implements and compares two foundational generative architectures:

- **VAE** (Kingma & Welling, 2013) — a probabilistic latent variable model
- **DCGAN** (Radford et al., 2015) — an adversarial generative model

Both models are trained on human face images (CelebA dataset, 64×64 resolution) and evaluated on image quality, diversity, reconstruction capability, and training stability.

---

## 2. Background Theory

### 2.1 Variational Autoencoder (VAE)

A VAE is a generative latent-variable model that learns:

1. **Encoder** q_φ(z|x): maps input x to a distribution over latent space z  
2. **Decoder** p_θ(x|z): maps latent vector z back to image space

The key innovation is the **reparameterization trick**:

```
z = μ + ε · σ,   where ε ~ N(0, I)
```

This makes the sampling operation differentiable so gradients can flow through the stochastic node.

**Training objective — ELBO (Evidence Lower Bound):**

```
L = E[log p_θ(x|z)] - β · KL(q_φ(z|x) ‖ p(z))
  = Reconstruction Term   -  β · KL Regularisation
```

- **Reconstruction term**: How well the decoder recreates the input (MSE loss)
- **KL term**: Forces the latent distribution to stay close to N(0, I), creating a smooth, continuous latent space
- **β parameter**: Controls the trade-off (β-VAE: higher β → more disentangled representations)

### 2.2 Deep Convolutional GAN (DCGAN)

A GAN trains two networks in competition:

- **Generator G**: Maps random noise z ~ N(0, I) → synthetic image
- **Discriminator D**: Classifies images as real or fake

**Minimax objective:**

```
min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
```

In practice, we use the **non-saturating generator loss**:

```
L_G = -E[log D(G(z))]     (maximise log-probability of fooling D)
L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
```

DCGAN improvements over vanilla GAN:
- Strided convolutions instead of pooling
- Batch normalisation in all layers except first discriminator and last generator
- LeakyReLU in discriminator; ReLU in generator
- Adam optimiser with β₁=0.5

---

## 3. Dataset

**Dataset:** CelebA (Large-scale CelebFaces Attributes)  
**Size:** 202,599 celebrity face images  
**Preprocessing:**
- Resize to 178×178 then centre-crop → 64×64
- Normalize to [-1, 1] (mean=0.5, std=0.5 per channel)
- 90/10 train/validation split

**Fallback:** CIFAR-10 (50,000 images, auto-downloaded) if CelebA is unavailable.

---

## 4. Model Architecture

### 4.1 VAE Architecture

| Component | Layer Details | Output Shape |
|---|---|---|
| Input | — | (3, 64, 64) |
| Encoder Conv1 | Conv(3→64, k=4, s=2, p=1) + BN + LReLU | (64, 32, 32) |
| Encoder Conv2 | Conv(64→128, k=4, s=2, p=1) + BN + LReLU | (128, 16, 16) |
| Encoder Conv3 | Conv(128→256, k=4, s=2, p=1) + BN + LReLU | (256, 8, 8) |
| Encoder Conv4 | Conv(256→512, k=4, s=2, p=1) + BN + LReLU | (512, 4, 4) |
| Encoder Conv5 | Conv(512→512, k=4, s=1, p=0) + LReLU | (512, 1, 1) |
| FC μ | Linear(512 → 128) | (128,) |
| FC log σ² | Linear(512 → 128) | (128,) |
| **Latent z** | **Reparameterize** | **(128,)** |
| Decoder FC | Linear(128 → 512) + Reshape | (512, 1, 1) |
| Decoder TConv1 | ConvT(512→256, k=4, s=1) + BN + ReLU | (256, 4, 4) |
| Decoder TConv2 | ConvT(256→128, k=4, s=2) + BN + ReLU | (128, 8, 8) |
| Decoder TConv3 | ConvT(128→64, k=4, s=2) + BN + ReLU | (64, 16, 16) |
| Decoder TConv4 | ConvT(64→32, k=4, s=2) + BN + ReLU | (32, 32, 32) |
| Decoder TConv5 | ConvT(32→3, k=4, s=2) + Tanh | (3, 64, 64) |

**Total parameters:** ~8.5M

### 4.2 DCGAN Architecture

**Generator:**

| Layer | Details | Output Shape |
|---|---|---|
| Input noise | z ~ N(0, I) | (100,) |
| ConvT1 | ConvT(100→512, k=4, s=1) + BN + ReLU | (512, 4, 4) |
| ConvT2 | ConvT(512→256, k=4, s=2) + BN + ReLU | (256, 8, 8) |
| ConvT3 | ConvT(256→128, k=4, s=2) + BN + ReLU | (128, 16, 16) |
| ConvT4 | ConvT(128→64, k=4, s=2) + BN + ReLU | (64, 32, 32) |
| ConvT5 | ConvT(64→3, k=4, s=2) + Tanh | (3, 64, 64) |

**Discriminator:**

| Layer | Details | Output Shape |
|---|---|---|
| Input | Image | (3, 64, 64) |
| Conv1 | Conv(3→64, k=4, s=2) + LReLU | (64, 32, 32) |
| Conv2 | Conv(64→128, k=4, s=2) + BN + LReLU | (128, 16, 16) |
| Conv3 | Conv(128→256, k=4, s=2) + BN + LReLU | (256, 8, 8) |
| Conv4 | Conv(256→512, k=4, s=2) + BN + LReLU | (512, 4, 4) |
| Conv5 | Conv(512→1, k=4, s=1) + Sigmoid | (1,) |

**Generator parameters:** ~3.5M  
**Discriminator parameters:** ~2.7M

---

## 5. Training Details

### 5.1 VAE Training

| Hyperparameter | Value |
|---|---|
| Epochs | 30 |
| Batch size | 128 |
| Optimizer | Adam (lr=1e-3) |
| LR scheduler | StepLR (×0.5 every 10 epochs) |
| β (KL weight) | 1.0 |
| Gradient clipping | max_norm=1.0 |
| Latent dim | 128 |

### 5.2 GAN Training

| Hyperparameter | Value |
|---|---|
| Epochs | 50 |
| Batch size | 128 |
| Optimizer G | Adam (lr=2e-4, β₁=0.5) |
| Optimizer D | Adam (lr=2e-4, β₁=0.5) |
| Label smoothing | real→0.9, fake→0.0 |
| Noise dim | 100 |
| Weight init | N(0, 0.02) for all Conv layers |

---

## 6. Results

> **Note:** The values below are *expected typical results* for these architectures on CelebA at 64×64. Actual numbers from your training run will appear in `outputs/plots/metrics.txt` after running `python evaluation/evaluate.py`.

### 6.1 VAE Results

| Metric | Expected Value |
|---|---|
| Reconstruction MSE | 0.015 – 0.040 |
| PSNR | 20 – 26 dB |
| SSIM | 0.55 – 0.78 |
| Diversity Score | 0.10 – 0.18 |

**Observation:** VAE reconstructions are recognisably similar to inputs but noticeably blurry. Generated samples from the prior are diverse but lack sharpness.

### 6.2 GAN Results

| Metric | Expected Value |
|---|---|
| G Loss (final) | 2.0 – 4.0 |
| D Loss (final) | 0.5 – 1.2 |
| D(real) score | ~0.6 – 0.8 |
| D(fake) score | ~0.2 – 0.5 |
| Diversity Score | 0.12 – 0.22 |

**Observation:** GAN-generated faces are sharper with more visible high-frequency detail. Training may exhibit instability (oscillating losses) around epochs 15–30.

---

## 7. Model Comparison

| Criterion | VAE | DCGAN |
|---|---|---|
| **Image sharpness** | Low (blurry) | High (sharp) |
| **Image diversity** | High | Moderate (risk of collapse) |
| **Reconstruction** | Yes | No |
| **Latent space** | Structured, continuous | Unstructured |
| **Training stability** | High | Moderate |
| **Convergence** | Smooth loss curves | Oscillating losses |
| **Mode collapse** | No | Possible |
| **Interpretability** | High (μ, σ interpretable) | Low |
| **Inference speed** | Fast | Fast |
| **Computational cost** | Moderate | Higher |

### Why do they differ?

**VAE** uses pixel-wise MSE as the reconstruction loss. MSE is minimised by predicting the **conditional mean** E[x|z]. When multiple valid reconstructions exist, their average appears blurry. The KL term forces all latent codes into a compact, smooth region of N(0,I).

**GAN** has no explicit pixel loss. Instead, the discriminator acts as a **learned perceptual loss**. It has implicitly learned what "real faces" look like and rejects blurry fakes. This adversarial pressure forces the generator to produce sharp, high-frequency features. The cost is training instability and potential mode collapse.

---

## 8. Failure Cases

### VAE Failure Cases
- **Blurriness**: Fundamental limitation of MSE-based reconstruction
- **Posterior collapse**: With very high β, the encoder may ignore the input and produce random samples (KL term dominates)
- **Out-of-distribution samples**: Sampling far from the origin in latent space produces distorted faces

### GAN Failure Cases
- **Mode collapse**: Generator produces identical (or near-identical) outputs regardless of z
- **Training instability**: D/G loss oscillations; one network dominates the other
- **Checkerboard artefacts**: From transposed convolution upsampling
- **Non-convergence**: GAN may never reach Nash equilibrium

---

## 9. Conclusions

1. **VAE** is better for applications requiring **reconstruction, interpolation, and a structured latent space** (e.g., style transfer, attribute manipulation, anomaly detection)
2. **GAN** is better for **maximum visual quality** when reconstruction is not required
3. Both models have fundamentally different failure modes
4. Modern models (VAE-GAN, VQ-VAE-2, StyleGAN) combine ideas from both to achieve high quality and stable training

---

## 10. Viva Questions and Answers

**Q1: What is the reparameterization trick and why is it needed?**  
A: Backpropagation cannot pass through a random sampling operation because sampling is not differentiable. The reparameterization trick rewrites z = μ + ε·σ where ε ~ N(0,I). Now z is a deterministic function of μ and σ (both learnable), with randomness isolated in ε. Gradients flow normally through μ and σ.

**Q2: What does the KL divergence term in VAE loss do?**  
A: It regularises the latent space by penalising distributions q(z|x) that deviate from the standard normal prior p(z) = N(0,I). Without it, the encoder would collapse to a delta function (σ→0, deterministic), creating "holes" in the latent space where decoding produces garbage. The KL term forces a smooth, continuous latent space from which we can sample during generation.

**Q3: Why are VAE outputs blurry?**  
A: MSE/L2 reconstruction loss is minimised by the conditional mean E[x|z]. When there is uncertainty about fine details, the model averages over all plausible pixel values, producing blur. A blurry average has lower total MSE than any sharp prediction when the model is uncertain.

**Q4: Explain the GAN training process.**  
A: Training alternates between two steps. (A) Discriminator update: Show D real images (label=1) and fake images from G (label=0); minimise BCE loss. (B) Generator update: Generate fake images, pass to D, compute BCE loss against label=1 (we want D to think they're real). This creates a minimax game where G tries to fool D while D tries to distinguish real from fake.

**Q5: What is mode collapse in GANs and how do you detect it?**  
A: Mode collapse is when the generator learns to produce only a few distinct outputs (or even one) that reliably fool the discriminator, ignoring most of the real data distribution. Detection: (a) Generated samples look nearly identical, (b) Diversity score drops sharply, (c) D(fake) score rises close to 0.5 while G loss drops — G has found a single "safe" output. Solutions include minibatch discrimination, spectral normalisation, or Wasserstein GAN.

**Q6: What is label smoothing and why is it used in GAN training?**  
A: Instead of training D with hard labels (1.0 for real, 0.0 for fake), we use soft labels (0.9 for real, 0.0 for fake). This prevents D from becoming overconfident, keeps gradients non-zero for the generator, and empirically improves training stability.

**Q7: What is the non-saturating generator loss?**  
A: In the original GAN formulation, L_G = log(1 - D(G(z))). Early in training D(G(z)) ≈ 0, so log(1-D(G(z))) ≈ log(1) = 0 — near-zero gradient for G. The non-saturating version flips the sign: L_G = -log(D(G(z))). Early gradient is now -1/D(G(z)) which is large when D(G(z)) is small. Same equilibrium, much stronger early learning signal.

**Q8: What metrics would you use to evaluate a generative model?**  
A: (1) MSE/PSNR/SSIM — for reconstruction quality (VAE only). (2) FID (Fréchet Inception Distance) — compares statistics of InceptionV3 features between real and generated distributions; lower is better. (3) IS (Inception Score) — measures quality and diversity of generated images. (4) Diversity score — std-dev across generated samples. (5) Human evaluation — the ultimate perceptual test.

**Q9: What is β-VAE?**  
A: β-VAE modifies the VAE loss to L = Recon + β·KL with β > 1. Higher β forces stronger disentanglement — individual latent dimensions tend to capture independent generative factors (e.g., one dimension for pose, another for lighting). The trade-off is worse reconstruction quality.

**Q10: How would you improve these models?**  
A: VAE: use perceptual loss (VGG features) instead of MSE to reduce blurriness; use VQ-VAE for discrete latent codes. GAN: use spectral normalisation, progressive growing (ProGAN), style-based generator (StyleGAN), or Wasserstein loss (WGAN-GP). Both: increase model capacity, train longer, use more data.

---

## 11. 2-Minute Presentation Script

*"Our project implements two generative deep-learning models — a VAE and a DCGAN — from scratch in PyTorch, trained on the CelebA face dataset at 64×64 resolution.*

*The VAE works as an encoder-decoder. The encoder maps each image to a mean and variance in a 128-dimensional latent space. We then apply the reparameterization trick — sampling z = μ + ε·σ — to make this stochastic step differentiable. The decoder reconstructs the image from z. We train it with MSE reconstruction loss plus a KL divergence term that regularises the latent space to look like a standard normal distribution. This lets us generate new faces by sampling random z vectors.*

*The DCGAN trains two networks in competition. The Generator takes 100-dimensional random noise and uses four transposed convolution layers to produce 64×64 face images. The Discriminator uses strided convolutions to classify images as real or fake. They play a minimax game — D tries to distinguish real from fake, G tries to fool D. We use the non-saturating generator loss and label smoothing for stability.*

*The key difference is: VAE uses pixel-wise MSE loss which averages over uncertainty, producing smooth but blurry images. GAN uses an adversarial loss — the discriminator acts as a learned perceptual judge — which forces G to produce sharp, realistic details.*

*We evaluate both models using PSNR and SSIM for VAE reconstruction, diversity score, and pixel-space FID as a quality metric. VAE gives stable training and structured latent space with interpolatable representations. GAN gives sharper images but risks mode collapse and training instability.*

*The complete project includes training scripts, evaluation pipeline, and an interactive Streamlit app where you can generate faces from both models and compare results."*

---

## 12. References

1. Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *arXiv:1312.6114*
2. Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. *arXiv:1511.06434*
3. Liu, Z., et al. (2015). Deep Learning Face Attributes in the Wild. *ICCV 2015*
4. Heusel, M., et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID). *NeurIPS 2017*
5. Higgins, I., et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. *ICLR 2017*
6. Goodfellow, I., et al. (2014). Generative Adversarial Networks. *NeurIPS 2014*
