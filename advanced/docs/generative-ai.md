# Generative AI

> 📍 [Home](../../README.md) › [Advanced](../README.md) › Generative AI

---

## Learning Objectives

By the end of this document you will be able to:

- Explain the key families of generative models (GANs, VAEs, diffusion, autoregressive)
- Describe how diffusion models generate images and their advantages over GANs
- Understand how CLIP and contrastive learning connect text and image representations
- Explain multimodal models that handle both vision and language inputs/outputs
- Trace how each modern generative architecture builds on foundational neural network concepts

---

## Prerequisites

| Topic | Where to Review |
|-------|----------------|
| Feedforward networks and backpropagation | [Neural Networks](neural-networks.md) |
| Transformer architecture | [Model Architectures](model-architectures.md) |
| LLM training and inference | [Large Language Models](large-language-models.md) |

---

## 1. What Is Generative AI?

Generative AI refers to models that learn the underlying distribution of a dataset and can **sample new instances** from that distribution. Given training data `x ~ p_data(x)`, a generative model learns an approximation `p_model(x) ≈ p_data(x)`.

Unlike discriminative models (which learn `P(y|x)` — the label given input), generative models learn `P(x)` or `P(x|c)` for conditioning variable `c` (e.g., a text prompt).

### The Four Major Generative Paradigms

| Paradigm | Core Idea | Key Models |
|----------|-----------|-----------|
| **Autoregressive** | Model `P(x)` as a chain of conditionals `Π P(x_t | x_{<t})` | GPT, LLaMA, AudioLM |
| **Variational Autoencoder (VAE)** | Encode to latent distribution; decode by sampling | VAE, VQ-VAE, Stable Diffusion's autoencoder |
| **Generative Adversarial Network (GAN)** | Generator vs discriminator adversarial game | StyleGAN, BigGAN, CycleGAN |
| **Diffusion Model** | Learn to reverse a gradual noising process | DALL-E 3, Stable Diffusion, Sora |

---

## 2. Variational Autoencoders (VAEs)

A VAE consists of an **encoder** that maps input `x` to a distribution over a latent space `z`, and a **decoder** that reconstructs `x` from samples of `z`.

### The VAE Objective

Standard autoencoders learn a deterministic latent code — VAEs instead encode to a **distribution** `q_φ(z|x) = N(μ, σ²)`:

```
ELBO = E_{z~q_φ(z|x)} [log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
         ─────────────────────────────   ─────────────────────────
              Reconstruction loss              Regularisation term
              (how well we can                (keep latent near
               reconstruct x from z)           standard normal)
```

**Reparameterisation trick:** Sampling `z = μ + σ·ε` where `ε ~ N(0,1)` allows gradients to flow through the stochastic sampling step.

```python
import torch
import torch.nn as nn

class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 256)
        self.mu = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor):
        h = torch.relu(self.fc(x))
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var

    def reparameterise(self, mu, log_var):
        # Reparameterisation trick: z = mu + sigma * epsilon
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
```

### VQ-VAE: Discrete Latent Spaces

Vector Quantised VAE (VQ-VAE) replaces the continuous latent space with a **discrete codebook** — the encoder output is mapped to the nearest code vector. This enables:
- Discrete tokens suitable for autoregressive modelling (e.g., generating images as token sequences)
- Lossless reconstruction through quantisation
- The foundation for models like DALL-E (v1), AudioLM, and MusicLM

---

## 3. Generative Adversarial Networks (GANs)

Introduced by Goodfellow et al. (2014), GANs pit two networks against each other:

- **Generator `G`:** Takes a random noise vector `z ~ N(0, I)` and produces a synthetic sample `G(z)`
- **Discriminator `D`:** Classifies whether an input is real (from training data) or fake (from `G`)

### The Minimax Objective

```
min_G max_D E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]
```

At equilibrium, `G(z)` is indistinguishable from real data and `D` outputs 0.5 everywhere.

```python
# Simplified GAN training loop
for real_images, _ in dataloader:
    batch_size = real_images.size(0)

    # ── Train Discriminator ──
    noise = torch.randn(batch_size, latent_dim)
    fake_images = generator(noise).detach()

    real_loss = criterion(discriminator(real_images), torch.ones(batch_size, 1))
    fake_loss = criterion(discriminator(fake_images), torch.zeros(batch_size, 1))
    d_loss = (real_loss + fake_loss) / 2

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # ── Train Generator ──
    noise = torch.randn(batch_size, latent_dim)
    gen_images = generator(noise)
    g_loss = criterion(discriminator(gen_images), torch.ones(batch_size, 1))

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
```

### GAN Training Challenges

| Challenge | Description | Mitigation |
|-----------|-------------|-----------|
| **Mode collapse** | Generator produces only a few output modes | Unrolled GANs, Wasserstein GANs |
| **Training instability** | Loss oscillates, fails to converge | Gradient penalty (WGAN-GP), spectral normalisation |
| **Discriminator dominance** | D becomes too strong; G receives no gradient | One-sided label smoothing, learning rate balance |

### Notable GAN Architectures

**StyleGAN (Karras et al., 2019):** Introduced style-based generator with:
- Mapping network that maps latent `z` to intermediate latent `w`
- Adaptive instance normalisation (AdaIN) to inject style at each resolution
- Generates photorealistic human faces at 1024×1024

**Conditional GAN (cGAN):** Conditions both G and D on class label or text embedding, enabling controlled generation.

**CycleGAN:** Unpaired image-to-image translation (e.g., photo → painting) using cycle consistency loss.

---

## 4. Diffusion Models

Diffusion models have largely superseded GANs for high-quality image generation due to their training stability and mode coverage.

### The Forward Process (Adding Noise)

A diffusion model defines a **fixed** forward Markov chain that gradually adds Gaussian noise to data over `T` timesteps (typically T=1000):

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) · x_{t-1}, β_t · I)
```

Where `β_t` is the noise schedule. By timestep `T`, `x_T ≈ N(0, I)` — pure noise.

Using the reparameterisation: `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε` where `ε ~ N(0, I)` and `ᾱ_t = Π_{s=1}^{t} (1-β_s)`.

### The Reverse Process (Denoising)

The model learns the **reverse process** — how to remove noise step by step to recover the original image:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

In practice, the neural network `ε_θ(x_t, t)` is trained to **predict the noise** added at each step:

```
L = E_{t, x_0, ε} [ ||ε - ε_θ(√ᾱ_t · x_0 + √(1-ᾱ_t) · ε, t)||² ]
```

```python
# Simplified diffusion training step
def train_step(model, x_0, timestep, noise_scheduler):
    # Sample random noise
    noise = torch.randn_like(x_0)

    # Add noise to image at timestep t (forward process)
    x_t = noise_scheduler.add_noise(x_0, noise, timestep)

    # Predict the noise that was added (the denoising network)
    predicted_noise = model(x_t, timestep)

    # Train to accurately predict the noise
    loss = F.mse_loss(predicted_noise, noise)
    return loss
```

### The U-Net Denoising Architecture

The neural network backbone for diffusion models is typically a **U-Net** with:
- Encoder path: downsamples spatial resolution, increases channels
- Decoder path: upsamples back to original resolution (with skip connections from encoder)
- **Attention layers** inserted at intermediate resolutions to capture global context

For text-conditional generation, the text embedding is injected via **cross-attention** at each U-Net level.

### Latent Diffusion Models

Running diffusion in pixel space is computationally expensive. **Latent Diffusion Models (LDMs)** — the architecture behind Stable Diffusion — run diffusion in a **compressed latent space**:

```
┌───────────────────────────────────────────────────────┐
│  Text Prompt                                           │
│    ↓                                                   │
│  Text Encoder (CLIP / T5) → Text Embeddings           │
│                                       ↓ (cross-attn)  │
│  x_0 → VAE Encoder → z_0 → Diffusion → z_0'          │
│                                            ↓           │
│                               VAE Decoder → x_0'      │
└───────────────────────────────────────────────────────┘
```

| | Pixel Diffusion | Latent Diffusion |
|-|----------------|-----------------|
| Diffusion space | 512×512×3 pixels | 64×64×4 latents (8× compression) |
| U-Net input size | Large | Small (much faster) |
| Quality | Excellent | Excellent |
| Speed | Slow | Much faster |
| Examples | DALL-E 2, GLIDE | Stable Diffusion, DALL-E 3 |

### DDIM Sampling and Acceleration

Standard DDPM requires ~1000 denoising steps. **DDIM** (Denoising Diffusion Implicit Models) enables high-quality generation in 20–50 steps by using a deterministic (non-Markovian) reverse process.

Further acceleration methods include:
- **DPM-Solver++**: ODE-based solvers; ~10–15 steps with high quality
- **Consistency Models**: Few-step or even single-step generation
- **Flow Matching**: Alternative to diffusion; simpler trajectories, faster training

---

## 5. Text-to-Image Generation

Text-to-image models combine two capabilities:
1. **Understanding text:** Encoding semantic meaning from a prompt
2. **Generating images:** Producing coherent, detailed images

### CLIP: Connecting Text and Images

CLIP (Contrastive Language-Image Pre-training, Radford et al., 2021) trains a text encoder and image encoder jointly using **contrastive learning**:

- A batch of `N` (image, caption) pairs is assembled
- The text and image encoders produce embeddings
- The training objective: maximise cosine similarity of matched pairs, minimise unmatched pairs

```
             Text Encoder
"a cat       ─────────────→  [text_embed_1]
 on a mat"                                 \
                                            ↘ Similarity matrix (N×N)
"a dog       ─────────────→  [text_embed_2]   ↗ Diagonal = matched pairs
 playing"                                  /
                                          /
 [cat img] ──────────────→  [img_embed_1]
 [dog img] ──────────────→  [img_embed_2]
             Image Encoder

Objective: maximise similarity of (text_i, image_i) pairs
```

```python
from transformers import CLIPModel, CLIPProcessor
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Compute text-image similarity
images = [...]  # PIL images
texts = ["a photo of a cat", "a photo of a dog"]

inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
outputs = model(**inputs)

# logits_per_image: similarity of each image to each text
logits = outputs.logits_per_image
probs = logits.softmax(dim=1)
```

CLIP embeddings are used as:
- **Guidance signals** in diffusion models (guide generation toward the prompt)
- **Universal image-text similarity metrics** (CLIP score)
- **Feature extractors** for zero-shot classification

### Stable Diffusion Architecture

Stable Diffusion (Rombach et al., 2022) combines:
- **CLIP text encoder:** Maps the text prompt to conditioning embeddings
- **Variational Autoencoder (VAE):** Compresses images into a 64×64×4 latent space
- **U-Net with cross-attention:** Denoises in latent space, conditioned on text embeddings

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe(
    prompt="A sunset over mountain peaks, cinematic, golden hour",
    negative_prompt="blurry, low quality, oversaturated",
    num_inference_steps=30,  # Fewer steps = faster, potentially lower quality
    guidance_scale=7.5,      # Classifier-free guidance strength
).images[0]
```

**Classifier-Free Guidance (CFG):** At each denoising step, both a conditional and an unconditional prediction are made; the final prediction extrapolates between them:

```
ε̃ = ε_uncond + w · (ε_cond - ε_uncond)
```

Higher guidance scale `w` → image more faithful to prompt, less diverse.

### DALL-E 3

OpenAI's DALL-E 3 uses a cascade architecture with:
1. A **text encoder** (large T5-style model)
2. A prior model that generates CLIP image embeddings from text
3. A diffusion decoder that generates images from CLIP embeddings

DALL-E 3's key advance: training on **re-captioned** images where GPT-4 generates detailed captions for each image. This dramatically improves prompt adherence.

---

## 6. Text Generation and Autoregressive Models

Text generation via autoregressive LLMs is covered in depth in [Large Language Models](large-language-models.md). Key points:

- Models generate text by predicting the next token from a vocabulary distribution
- Controlled by sampling strategies (temperature, top-p, top-k)
- Modern LLMs (GPT-4, Claude, Gemini) generate highly coherent, contextually appropriate text
- Structured output (JSON, code) is increasingly reliable with model scale

### Specialised Text Generation Domains

| Domain | Key Models | Notes |
|--------|-----------|-------|
| **Code generation** | GitHub Copilot, CodeLlama, DeepSeek-Coder | Trained on large code corpora (GitHub, etc.) |
| **Creative writing** | Claude 3.5, GPT-4o | Long-range coherence, style control |
| **Summarisation** | BART, T5, LLMs with RAG | Encoder-decoder or long-context LLMs |
| **Translation** | NLLB, mBART, GPT-4 | Multilingual pre-training |
| **SQL generation** | Text-to-SQL fine-tuned models | Schema-conditioned structured output |

---

## 7. Audio Generation

Audio generation has followed a similar trajectory to image generation — from autoregressive models to diffusion.

### Autoregressive Audio

**WaveNet (DeepMind, 2016):** Generates raw audio waveforms autoregressively using dilated causal convolutions. Ground-breaking quality but extremely slow (one sample at a time at 16kHz).

**AudioLM (Google, 2022):** Hierarchical autoregressive model:
1. Tokenise audio using SoundStream (a VQ-VAE for audio)
2. Generate coarse semantic tokens with a large transformer
3. Generate fine acoustic tokens conditioned on semantic tokens

### Diffusion-Based Audio

**Stable Audio (Stability AI):** Adapts latent diffusion to audio spectrograms. Generates high-quality music and sound effects from text prompts.

**Voicebox (Meta, 2023):** Uses flow matching (a continuous diffusion variant) for zero-shot voice synthesis and style transfer.

---

## 8. Multimodal Models

Multimodal models process and generate across multiple modalities simultaneously.

### Vision-Language Models (VLMs)

VLMs accept image + text as input and produce text as output. Architecture families:

**Cross-attention injection:**
- Image features extracted by a vision encoder (e.g., CLIP ViT or SigLIP)
- Image features injected into the language model via cross-attention at every layer
- Used in: Flamingo, IDEFICS

**Projection layer (most common):**
- Visual encoder produces patch-level features
- A **projection layer** (linear or MLP) maps visual features into the LLM's embedding space
- Visual tokens concatenated with text tokens in the input sequence
- Used in: LLaVA, Qwen-VL, InternVL, PaliGemma

```
┌─────────────┐    ┌────────────────┐    ┌────────────────────┐
│ Input Image │ →  │ Vision Encoder │ →  │ Image Tokens       │
│             │    │ (ViT / SigLIP) │    │ (visual features)  │
└─────────────┘    └────────────────┘    └──────────┬─────────┘
                                                     │ concat
                                                     ↓
┌─────────────┐                          ┌────────────────────┐
│  "Describe  │ ───────────────────────→ │   Language Model   │
│  this image"│  Text token embeddings   │   (decoder-only)   │
└─────────────┘                          └────────────────────┘
```

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
).to("cuda")

image = Image.open("chart.png")
prompt = "[INST] <image>\nWhat are the key trends shown in this chart? [/INST]"

inputs = processor(prompt, image, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Native Multimodal Models

Frontier models like GPT-4o and Gemini 1.5 are trained natively on text, image, audio, and video jointly — not by bolting a vision encoder onto a language model. This enables:
- Direct image and audio generation alongside text
- More unified understanding across modalities
- Capabilities like real-time voice + vision interaction (GPT-4o demo)

**Gemini 1.5's architecture:** Uses a sparse MoE Transformer trained on interleaved text, image, audio, and video tokens from the start.

### Text-to-Video

Video generation is an active frontier, combining spatial (image) and temporal modelling.

**Sora (OpenAI, 2024):** Diffusion model that operates on **spatiotemporal patches** (video tokens). Key innovations:
- Treats video as sequences of spacetime patches (flexible resolution + duration)
- Scales transformer-based diffusion to video generation
- Generates videos up to 1 minute with coherent camera motion

**Runway Gen-3, Stability AI Stable Video Diffusion:** Adapt image diffusion models to video with temporal attention layers.

---

## 9. How Generative AI Builds on Foundational Concepts

Modern generative AI is built entirely on foundational neural network principles — applied at scale and combined in novel ways:

| Foundation | Generative AI Application |
|-----------|--------------------------|
| **Feedforward networks (MLPs)** | FFN sublayers in every transformer; decoder networks in VAEs/GANs |
| **Convolutional layers** | Image encoders/decoders, U-Net for diffusion |
| **Backpropagation** | Training all generative models via gradient descent |
| **Cross-entropy loss** | Training autoregressive text and image generation |
| **Attention mechanism** | Core of LLMs; U-Net cross-attention for text conditioning |
| **Transformers** | Backbone of LLMs, VLMs, and diffusion U-Nets |
| **Embeddings** | Token embeddings (text), patch embeddings (images), audio tokens |
| **Normalisation (BatchNorm, LayerNorm)** | Stabilise GAN training; Pre-Norm in transformers |
| **Latent spaces (VAEs)** | Compressed representations enabling latent diffusion |
| **Contrastive learning (CLIP)** | Bridges text and image modalities |

---

## Key Concepts Summary

| Concept | One-line description |
|---------|---------------------|
| **VAE** | Encoder maps to distribution; decoder samples and reconstructs |
| **Reparameterisation trick** | `z = μ + σε` — makes sampling differentiable |
| **GAN** | Generator and discriminator trained adversarially |
| **Mode collapse** | Generator produces limited output variety |
| **Diffusion model** | Learns to reverse a gradual noising process |
| **DDPM/DDIM** | Denoising diffusion probabilistic/implicit models; DDIM is faster |
| **Latent diffusion** | Run diffusion in VAE-compressed latent space (Stable Diffusion) |
| **CFG** | Classifier-free guidance — steers generation toward prompt |
| **CLIP** | Contrastive text-image pretraining; bridges modalities |
| **VLM** | Vision-Language Model; processes image + text inputs |
| **Sora** | Video diffusion via spatiotemporal patches |

---

## 🧠 Knowledge Check

Test your understanding. Attempt each question before revealing the answer.

**Q1.** In a diffusion model, what is the role of the forward process vs. the reverse process?

<details>
<summary>Answer</summary>

- **Forward process (noising):** A fixed, non-learned Markov chain that gradually adds Gaussian noise to a real data sample over T timesteps (e.g., T=1000) until the sample becomes approximately pure Gaussian noise. This process is analytically tractable.

- **Reverse process (denoising):** The learned component. Starting from pure noise, the model iteratively predicts and subtracts the noise added at each timestep to recover a coherent sample. This is the generative direction — what the neural network (typically a U-Net) learns.

Generation at inference time runs only the reverse process: sample pure noise, then denoise T times.

</details>

---

**Q2.** What is the key computational advantage of Latent Diffusion Models (like Stable Diffusion) over pixel-space diffusion models?

<details>
<summary>Answer</summary>

Latent Diffusion Models perform the diffusion process in a **compressed latent space** encoded by a pretrained VAE, rather than directly in pixel space. A 512×512 image is compressed to a ~64×64×4 latent representation — about 8× smaller in each spatial dimension.

This means the U-Net denoising network operates on much smaller tensors, reducing training and inference compute by orders of magnitude compared to pixel-space diffusion. The quality loss from the VAE compression is minimal for most applications. The VAE decoder then maps the denoised latent back to the full-resolution image.

</details>

---

**Q3.** How does CLIP enable zero-shot image classification without task-specific training data?

<details>
<summary>Answer</summary>

CLIP trains an image encoder and a text encoder jointly using a contrastive objective: for each image-text pair in a training batch, their embeddings should be similar (high cosine similarity); all non-matching pairs should be dissimilar.

For zero-shot classification: given a new image and candidate class labels, CLIP formats each label as a text template (e.g., "a photo of a {class}"), encodes the image and all text descriptions, and assigns the class whose text embedding has the highest cosine similarity to the image embedding. No task-specific training data or fine-tuning is required — the shared embedding space enables this direct comparison.

</details>

---

➡️ **Full quiz with 3 questions:** [Knowledge Checks → Generative AI](knowledge-checks.md#5-generative-ai)

---

## Further Reading

| Resource | Type | Notes |
|---------|------|-------|
| [Auto-Encoding Variational Bayes (VAE)](https://arxiv.org/abs/1312.6114) | Paper | Original VAE paper (Kingma & Welling, 2014) |
| [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) | Paper | Original GAN paper (Goodfellow et al., 2014) |
| [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) | Paper | Ho et al., 2020 — modern diffusion framework |
| [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | Paper | Stable Diffusion / LDM paper |
| [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) | Paper | OpenAI CLIP |
| [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) | Paper | Practical VLM construction |
| [Video Generation Models as World Simulators (Sora)](https://openai.com/research/video-generation-models-as-world-simulators) | Report | OpenAI's Sora technical report |
| [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) | Blog | Excellent visual walkthrough of LDMs |

---

*Navigation: [← Large Language Models](large-language-models.md) · [Advanced Home](../README.md) · [Next: Prompt Engineering →](prompt-engineering.md)*
