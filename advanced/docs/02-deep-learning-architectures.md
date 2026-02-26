# Deep Learning Architectures

> **Learning Objectives**
>
> By the end of this document, you will be able to:
> - Explain why convolutional operations are well-suited for spatial data and implement a CNN for image classification
> - Describe how recurrent neural networks process sequential data and identify the vanishing gradient problem
> - Explain how LSTM gates solve the long-range dependency problem that plain RNNs cannot handle
> - Implement a simple GAN and describe the adversarial training dynamics and their failure modes
> - Distinguish between autoencoders and variational autoencoders, and explain the reparameterization trick

---

## 1. Convolutional Neural Networks (CNNs)

### 1.1 Motivation: Exploiting Spatial Structure

Fully connected (dense) networks treat each input feature independently. For a 256×256 RGB image, a single dense layer connecting to 1,000 neurons would require 256 × 256 × 3 × 1,000 = **196 million parameters** — and it would learn nothing about the spatial structure of images: that nearby pixels are related, that the same edge detector is useful whether it appears at the top-left or bottom-right of an image.

CNNs address this through two key structural innovations:

1. **Local connectivity:** Each neuron connects only to a small spatial region (receptive field) of the input, not all input pixels.
2. **Weight sharing (parameter sharing):** A single set of filter weights is applied at every position in the input — the same edge detector works everywhere.

These constraints reduce parameters dramatically and encode **translation equivariance**: if a feature appears at any position in the input, the convolution detects it regardless of location.

### 1.2 The Convolution Operation

A **filter** (also called a kernel) is a small weight matrix, typically 3×3 or 5×5. It slides across the input, computing the dot product between the filter weights and the local input patch at each position. This produces a **feature map** (or activation map).

**For a single filter `F` of size `k×k` applied to input `X`:**

```
(X * F)[i, j] = Σₘ Σₙ X[i+m, j+n] · F[m, n]
```

A CNN layer typically applies `C_out` filters to produce `C_out` feature maps. An input with spatial dimensions `H × W` and `C_in` channels, processed by `C_out` filters of size `k×k`, produces output of shape `C_out × H' × W'`.

**Key hyperparameters:**
- **Filter size (`k`):** Typically 3×3 (small, expressive, memory-efficient) or 5×5. Larger filters have wider receptive fields but more parameters.
- **Stride (`s`):** How many pixels the filter moves per step. Stride 1 preserves spatial resolution; stride 2 halves it (subsampling).
- **Padding:** Adding zeros around the input border. "Same" padding (`p = (k-1)/2`) preserves input spatial dimensions; "valid" padding (no padding) reduces them.

**Output spatial dimensions:**

```
H' = floor((H + 2p - k) / s) + 1
W' = floor((W + 2p - k) / s) + 1
```

### 1.3 Pooling Layers

Pooling layers reduce spatial dimensions while retaining the most salient information. They also introduce a degree of **translation invariance** (slight shifts in input produce the same output).

**Max pooling:** Takes the maximum value in each non-overlapping `k×k` window. Most commonly 2×2 with stride 2, which halves height and width.

**Average pooling:** Takes the mean. Used in global average pooling (GAP) before classification heads, where each feature map is reduced to a single scalar — often replacing large dense layers.

```python
import torch
import torch.nn as nn

# 2×2 max pooling with stride 2
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Global average pooling: H×W → 1×1 per channel
gap = nn.AdaptiveAvgPool2d((1, 1))
```

### 1.4 A Simple CNN Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple CNN for 32×32 RGB image classification (10 classes, e.g., CIFAR-10).
    Architecture: Conv → BN → ReLU → Pool (×2) → Fully Connected
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Block 1: (3, 32, 32) → (32, 16, 16)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Block 2: (32, 16, 16) → (64, 8, 8)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Block 3: (64, 8, 8) → (128, 4, 4)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Classifier: 128×4×4 = 2048 → 10
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)

# Parameter count
model = SimpleCNN()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")    # ~1.3M
```

### 1.5 Classic Architectures

Understanding landmark CNN architectures reveals a progression of design innovations:

| Architecture | Year | Innovation | Top-1 ImageNet |
|---|---|---|---|
| **LeNet-5** | 1998 | Original CNN: conv + pool + dense | ~1% on MNIST |
| **AlexNet** | 2012 | Deep CNN on GPU; ReLU; Dropout; data augmentation | 63.3% |
| **VGG-16/19** | 2014 | Uniform 3×3 conv throughout; deep (16–19 layers) | 74.5% |
| **GoogLeNet (Inception)** | 2014 | Inception module: parallel branches with different filter sizes | 74.8% |
| **ResNet-50/152** | 2015 | Residual (skip) connections; trained 152 layers | 77.1% |

### 1.6 Residual Networks and Skip Connections

The key challenge in training very deep networks is the vanishing gradient: as depth increases, gradients become vanishingly small before reaching early layers. ResNet (He et al., 2015) solved this with the **residual block**:

```
y = F(x, {Wᵢ}) + x         (shortcut connection — adds input directly to output)
```

Instead of learning the target mapping `H(x)`, each block learns the **residual** `F(x) = H(x) - x`. If the optimal transformation is close to the identity, it is easier to push the residual toward zero than to reproduce the identity through a stack of layers.

```python
class ResidualBlock(nn.Module):
    """
    Standard ResNet bottleneck block.
    Input and output have the same channel count.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv_block(x) + x)   # Add skip connection before final activation
```

Skip connections ensure that gradients can flow directly from any layer to earlier layers, enabling training of 100+ layer networks. This principle has generalized far beyond ResNets — it is fundamental to transformers (residual connections around attention and feed-forward blocks) and many other modern architectures.

---

## 2. Recurrent Neural Networks (RNNs)

### 2.1 The Problem of Sequential Data

A feedforward network's input size is fixed. Sequences — text, audio, time series, video frames — vary in length. More importantly, the meaning of a word depends on the words that preceded it: "bank" means something different in "river bank" vs. "savings bank." An architecture for sequential data must maintain and update a **memory** of past inputs.

**Recurrent Neural Networks** process sequences by maintaining a **hidden state** `h_t` that is updated at each time step:

```
h_t = σ(W_hh · h_{t-1} + W_xh · x_t + b_h)       (hidden state update)
ŷ_t = W_hy · h_t + b_y                              (output at time t)
```

where:
- `x_t` — input at time step `t`
- `h_t` — hidden state (the network's "memory") at time `t`, initialized to `h_0 = 0`
- `W_hh, W_xh, W_hy` — learnable weight matrices **shared across all time steps**

The key insight is weight sharing across time — the same parameters `W_hh` and `W_xh` process every element of the sequence. This gives RNNs the ability to handle variable-length inputs.

### 2.2 Unrolled Computation Graph

An RNN can be unrolled in time, revealing it as a deep feedforward network where each "layer" corresponds to one time step:

```
x_1 → [h_1] → ŷ_1
         ↓
x_2 → [h_2] → ŷ_2
         ↓
x_3 → [h_3] → ŷ_3
```

The depth of this unrolled network equals the sequence length — often hundreds or thousands of steps for text.

### 2.3 Types of RNN Tasks

| Input/Output Pattern | Example | Architecture |
|---|---|---|
| **Many-to-one** | Sentiment classification | Output only at the final step |
| **Many-to-many (same length)** | Part-of-speech tagging | Output at every step |
| **Many-to-many (different length)** | Machine translation | Encoder-decoder RNN |
| **One-to-many** | Image captioning | Single input, sequence output |

### 2.4 Backpropagation Through Time (BPTT)

Training an RNN requires computing gradients across all time steps — called **Backpropagation Through Time (BPTT)**. The gradient of the loss at time `T` with respect to a weight used at time step `t` involves a product of Jacobians:

```
∂L_T/∂h_t = (∂L_T/∂h_T) · ∏_{k=t+1}^{T} ∂h_k/∂h_{k-1}
```

Since `∂h_k/∂h_{k-1} = diag(σ'(h_{k-1})) W_hh`, and `W_hh` is the same matrix at every step, the gradient involves `(W_hh)^(T-t)`. If the largest singular value of `W_hh`:
- **< 1:** The gradient shrinks exponentially with `T-t` — **vanishing gradient**
- **> 1:** The gradient grows exponentially — **exploding gradient**

In practice, vanishing gradients make plain RNNs unable to capture dependencies spanning more than ~10–20 time steps. Exploding gradients are addressed by **gradient clipping** (clamping gradient norm to a maximum value).

```python
# Gradient clipping in PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 2.5 PyTorch RNN

```python
import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    """Many-to-one RNN for text classification."""
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,       # (batch, seq_len, features) convention
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len)
        embeds = self.embedding(token_ids)          # (batch, seq_len, embed_dim)
        _, h_n = self.rnn(embeds)                   # h_n: (num_layers, batch, hidden)
        last_hidden = h_n[-1]                       # Take top layer's hidden state
        return self.classifier(last_hidden).squeeze(-1)
```

---

## 3. Long Short-Term Memory (LSTM) and GRU

### 3.1 The LSTM Architecture

The LSTM (Hochreiter & Schmidhuber, 1997) solves the vanishing gradient problem by introducing a **cell state** `C_t` — a "memory conveyor belt" that runs through the sequence with only small, multiplicative modifications at each step. A set of learned **gates** controls what information is written to, read from, and erased from this cell state.

**LSTM equations at time step `t`:**

```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)       (input gate:  what new info to write)
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)       (forget gate: what old info to erase)
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)   (candidate:   new candidate values)
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)      (output gate: what to reveal as h_t)

C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t          (cell state update)
h_t = o_t ⊙ tanh(C_t)                     (hidden state output)
```

where `⊙` is elementwise multiplication and `[h_{t-1}, x_t]` denotes concatenation.

**Understanding the gates:**

- **Forget gate `f_t`:** Values near 1 preserve existing cell state; values near 0 erase it. "Forget" the previous entity subject when encountering a new one in a text.
- **Input gate `i_t`:** Controls which candidate values `g_t` are actually written to the cell state.
- **Cell state update:** A combination of forgetting old memories and writing new ones.
- **Output gate `o_t`:** Controls which parts of the cell state to expose as the hidden state `h_t`.

The cell state update `C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t` is a **linear recurrence** (from the perspective of gradient flow): the gradient can flow back through `f_t ⊙ C_{t-1}` with much less attenuation than through a nonlinear hidden state update. This is why LSTMs can learn dependencies over hundreds of time steps.

### 3.2 PyTorch LSTM

```python
class LSTMLanguageModel(nn.Module):
    """
    Character-level or token-level language model using LSTM.
    Predicts the next token in a sequence.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, state=None) -> tuple:
        """
        x: (batch, seq_len) token indices
        state: (h_0, c_0) from previous batch (for truncated BPTT)
        """
        embeds = self.embedding(x)                      # (batch, seq, embed_dim)
        outputs, (h_n, c_n) = self.lstm(embeds, state) # outputs: (batch, seq, hidden)
        logits = self.output_proj(outputs)              # (batch, seq, vocab_size)
        return logits, (h_n.detach(), c_n.detach())    # Detach for truncated BPTT

    def generate(self, seed_tokens: list[int], max_new_tokens: int, temperature: float = 1.0) -> list[int]:
        """Autoregressive generation from a seed sequence."""
        self.eval()
        generated = list(seed_tokens)
        x = torch.tensor(seed_tokens).unsqueeze(0)     # (1, len(seed))
        state = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, state = self.forward(x, state)
                last_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)
                x = torch.tensor([[next_token]])        # Feed back as next input

        return generated
```

### 3.3 Gated Recurrent Unit (GRU)

The GRU (Cho et al., 2014) is a simplified variant of the LSTM that merges the cell state and hidden state and uses two gates instead of three:

```
z_t = σ(W_z · [h_{t-1}, x_t])           (update gate: how much to update h)
r_t = σ(W_r · [h_{t-1}, x_t])           (reset gate: how much past to consider)
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])   (candidate hidden state)
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  (final hidden state)
```

**LSTM vs. GRU trade-offs:**

| Aspect | LSTM | GRU |
|---|---|---|
| Parameters | ~4x hidden_dim² per layer | ~3x hidden_dim² per layer |
| Expressivity | Higher (separate cell state) | Slightly lower |
| Training speed | Slower | ~25–33% faster |
| Performance | Marginal advantage on some tasks | Competitive on most tasks |
| **Recommendation** | Default choice for complex tasks | When speed/memory matters |

```python
# Swapping LSTM for GRU requires only one line change
self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim,
                   num_layers=num_layers, batch_first=True, dropout=0.3)
# GRU returns (output, h_n) — no separate cell state
outputs, h_n = self.gru(embeds)
```

### 3.4 Bidirectional RNNs

Standard RNNs are causal — `h_t` depends only on inputs up to time `t`. For tasks where the full context is available (e.g., text classification, named entity recognition — not generation), **bidirectional RNNs** process the sequence in both directions and concatenate the hidden states:

```
→h_t = RNN_forward(x_1, ..., x_t)         (left-to-right)
←h_t = RNN_backward(x_T, ..., x_t)        (right-to-left)
h_t = concat(→h_t, ←h_t)                  (doubled hidden size)
```

```python
self.bilstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
# Output size: 2 × hidden_dim (forward + backward)
```

---

## 4. Generative Adversarial Networks (GANs)

### 4.1 The Adversarial Framework

GANs (Goodfellow et al., 2014) train two networks in competition:

- **Generator `G`:** Takes random noise `z ~ p_z(z)` (typically Gaussian or uniform) as input and outputs synthetic data `G(z)`. Its goal is to produce samples indistinguishable from real data.
- **Discriminator `D`:** Takes a data sample (real or generated) as input and outputs a scalar probability that the sample is real. Its goal is to correctly classify real vs. fake.

**The minimax game:**

```
min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]
```

- The discriminator maximizes `V`: reward for correctly identifying real samples (`log D(x)`) and fake samples (`log(1 - D(G(z)))`).
- The generator minimizes `V`: tries to fool the discriminator into assigning high probability to generated samples.

At the Nash equilibrium, `G` produces samples that perfectly match `p_data` and `D(x) = 0.5` everywhere (the discriminator cannot distinguish real from fake).

### 4.2 Training Procedure

GANs alternate between updating `D` and `G`:

```python
import torch
import torch.nn as nn

def train_gan_step(
    generator: nn.Module,
    discriminator: nn.Module,
    real_images: torch.Tensor,
    optim_G: torch.optim.Optimizer,
    optim_D: torch.optim.Optimizer,
    latent_dim: int,
    device: str = "cuda",
) -> dict[str, float]:
    """One training step: update D once, then update G once."""
    batch_size = real_images.size(0)
    real_label = torch.ones(batch_size, 1, device=device)
    fake_label = torch.zeros(batch_size, 1, device=device)
    criterion = nn.BCEWithLogitsLoss()

    # ── Step 1: Train Discriminator ──────────────────────────────
    optim_D.zero_grad()

    # Real images: D should output 1
    real_pred = discriminator(real_images)
    loss_D_real = criterion(real_pred, real_label)

    # Fake images: D should output 0
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_images = generator(z).detach()   # detach: no gradients flow to G here
    fake_pred = discriminator(fake_images)
    loss_D_fake = criterion(fake_pred, fake_label)

    loss_D = loss_D_real + loss_D_fake
    loss_D.backward()
    optim_D.step()

    # ── Step 2: Train Generator ──────────────────────────────────
    optim_G.zero_grad()

    # G wants D to predict "real" (1) for generated images
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_images = generator(z)
    fake_pred = discriminator(fake_images)
    loss_G = criterion(fake_pred, real_label)   # Generator's loss: fool D

    loss_G.backward()
    optim_G.step()

    return {"loss_D": loss_D.item(), "loss_G": loss_G.item()}
```

**Non-saturating loss for the generator:** The original minimax formulation `log(1 - D(G(z)))` saturates (produces near-zero gradients) early in training when the discriminator is strong and `D(G(z)) ≈ 0`. Instead, train the generator to maximize `log D(G(z))`:

```python
# Non-saturating generator loss (preferred in practice)
loss_G = criterion(fake_pred, real_label)   # Maximize log D(G(z)) rather than minimize log(1-D(G(z)))
```

### 4.3 DCGAN Architecture

Deep Convolutional GAN (Radford et al., 2015) established architectural guidelines for stable GAN training on images:

```python
class DCGANGenerator(nn.Module):
    """Generator for 64×64 image generation."""
    def __init__(self, latent_dim: int = 100, channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),  # → (512, 4, 4)
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),          # → (256, 8, 8)
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),          # → (128, 16, 16)
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),           # → (64, 32, 32)
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),      # → (C, 64, 64)
            nn.Tanh(),   # Output in [-1, 1]; normalize images to this range
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.size(0), -1, 1, 1))

class DCGANDiscriminator(nn.Module):
    """Discriminator for 64×64 images."""
    def __init__(self, channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (C, 64, 64)
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),   # → (64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),         # → (128, 16, 16)
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),        # → (256, 8, 8)
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),        # → (512, 4, 4)
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),          # → (1, 1, 1)
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

**DCGAN guidelines:**
- Use strided convolutions (not pooling) for downsampling in D; transposed convolutions for upsampling in G
- Use Batch Normalization in both G and D (except the output of G and input of D)
- Use ReLU in G (except output layer, which uses Tanh); LeakyReLU in D

### 4.4 Training Challenges and Failure Modes

GAN training is notoriously unstable because the two networks must converge simultaneously to a Nash equilibrium — but are optimized with gradient descent, which is designed for minimization, not finding equilibria.

**Mode collapse:** The generator learns to produce a small variety of outputs (or even a single output) that reliably fools the current discriminator, rather than capturing the full diversity of the data distribution. The discriminator then adapts to reject this limited output, the generator shifts to another mode, and the cycle repeats.

*Symptoms:* Generated samples all look alike; loss oscillates dramatically.

*Mitigations:*
- **Minibatch discrimination:** Add features to D based on similarity between samples in a batch, penalizing low-diversity outputs
- **Wasserstein GAN (WGAN):** Replaces the JS divergence loss with the Wasserstein distance, providing smoother gradients and more stable training even when the generator is weak
- **Spectral normalization:** Normalizes weight matrices to enforce Lipschitz continuity on D, stabilizing training

**Training imbalance:** If D is too strong early in training, G gets near-zero gradients (saturating loss). If G is too strong, D can't provide useful signal.

*Mitigation:* Update D multiple times per G update (or vice versa); use progressive growing (ProGAN).

**Evaluation:** The **Fréchet Inception Distance (FID)** is the standard metric for GAN quality. It measures the Fréchet distance between the feature distributions (from a pretrained Inception network) of real and generated images — lower is better.

---

## 5. Autoencoders and Variational Autoencoders (VAEs)

### 5.1 Autoencoders: Learning Compact Representations

An autoencoder learns to compress input data into a lower-dimensional **latent representation** (encoding), then reconstruct the original input from that representation (decoding). By forcing information through a bottleneck, the model must learn the most important features of the data.

```
Input x → [Encoder fᵩ] → Latent code z = fᵩ(x) → [Decoder g_θ] → Reconstruction x̂ = g_θ(z)
```

**Training objective (reconstruction loss):**

For continuous data: `L = ||x - x̂||²` (MSE)
For binary/Bernoulli data: `L = -Σᵢ [xᵢ log(x̂ᵢ) + (1-xᵢ) log(1-x̂ᵢ)]` (binary cross-entropy)

```python
class Autoencoder(nn.Module):
    """Autoencoder for 28×28 grayscale images (e.g., MNIST)."""
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),   # Output in [0, 1] to match image pixel range
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)
```

**Applications:**
- **Dimensionality reduction:** Like PCA but nonlinear; useful for visualization and downstream ML
- **Anomaly detection:** Train on normal data; high reconstruction error signals anomalies
- **Denoising autoencoders:** Add noise to inputs during training; learn to reconstruct clean outputs

**Limitation:** The latent space of a standard autoencoder has no enforced structure — nearby points in latent space may not decode to similar outputs, and large regions of latent space may decode to garbage. This makes standard autoencoders poor generative models.

### 5.2 Variational Autoencoders (VAEs)

The VAE (Kingma & Welling, 2013) addresses this limitation by learning a **probabilistic latent space**: instead of mapping each input to a single point `z`, the encoder maps each input to a distribution `q_ᵩ(z|x)` — typically a Gaussian with learned mean `μ` and variance `σ²`:

```
Encoder: μ, log σ² = fᵩ(x)
z ~ N(μ, σ²)                      (sample from the distribution)
Decoder: x̂ = g_θ(z)
```

### 5.3 The ELBO Objective

The VAE is trained to maximize the **Evidence Lower BOund (ELBO)**:

```
ELBO = E_{q(z|x)}[log p(x|z)] - KL[q(z|x) || p(z)]
```

- **Reconstruction term** `E[log p(x|z)]`: Encourages the decoder to reconstruct the input faithfully (same as an autoencoder's reconstruction loss)
- **KL divergence term** `-KL[q(z|x) || p(z)]`: Forces the posterior distribution `q(z|x)` to stay close to the prior `p(z) = N(0, I)`, regularizing the latent space

**Closed form KL for Gaussians:**

```
KL[N(μ, σ²) || N(0, I)] = -0.5 × Σⱼ (1 + log σⱼ² - μⱼ² - σⱼ²)
```

### 5.4 The Reparameterization Trick

The sampling step `z ~ N(μ, σ²)` is not differentiable — gradients cannot flow through a random sampling operation. The reparameterization trick reformulates the sampling:

```
z = μ + σ ⊙ ε,        where ε ~ N(0, I)
```

Now `z` is a deterministic function of `μ` and `σ` (plus a fixed random sample `ε`), so gradients can flow through `z` to the encoder parameters. The randomness is "pushed" into `ε`, which does not depend on any parameters.

### 5.5 PyTorch VAE

```python
class VAE(nn.Module):
    """Variational Autoencoder for 28×28 grayscale images."""
    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: x → (μ, log σ²)
        self.encoder_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder: z → x̂
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_net(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)       # σ = exp(0.5 × log σ²)
        eps = torch.randn_like(std)          # ε ~ N(0, I)
        return mu + eps * std               # z = μ + σ ⊙ ε

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> tuple:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

def vae_loss(x_hat: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> torch.Tensor:
    """
    VAE loss = reconstruction loss + β × KL divergence.
    β > 1 (β-VAE) enforces more disentangled representations.
    """
    # Binary cross-entropy reconstruction loss
    recon_loss = nn.functional.binary_cross_entropy(
        x_hat.view(-1, 784), x.view(-1, 784), reduction="sum"
    )
    # KL divergence: -0.5 × Σ(1 + log σ² - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss
```

### 5.6 VAE Latent Space Properties

Because the KL term regularizes the posterior toward `N(0, I)`, the VAE latent space has useful geometric properties:

- **Continuity:** Nearby points in latent space decode to similar outputs — smooth interpolation is possible
- **Completeness:** Points sampled from `N(0, I)` decode to meaningful outputs (not garbage)
- **Sampling:** New data can be generated by sampling `z ~ N(0, I)` and decoding

```python
def generate_samples(vae: VAE, num_samples: int = 16) -> torch.Tensor:
    """Generate new images by sampling from the prior."""
    z = torch.randn(num_samples, vae.latent_dim)
    with torch.no_grad():
        return vae.decode(z)

def interpolate(vae: VAE, x1: torch.Tensor, x2: torch.Tensor, steps: int = 10) -> list:
    """Interpolate between two images in latent space."""
    mu1, _ = vae.encode(x1)
    mu2, _ = vae.encode(x2)
    alphas = torch.linspace(0, 1, steps)
    return [vae.decode((1 - a) * mu1 + a * mu2) for a in alphas]
```

### 5.7 VAE vs. GAN

| Aspect | VAE | GAN |
|---|---|---|
| **Training stability** | Stable (single loss, no adversarial dynamics) | Unstable; requires tuning |
| **Sample quality** | Good; often blurry due to pixel-wise MSE | High — often photo-realistic |
| **Latent space** | Structured, continuous, interpretable | Less structured; no direct encoder |
| **Mode coverage** | Good — posterior covers all modes | Risk of mode collapse |
| **Evaluation** | ELBO is a principled objective | FID; no direct likelihood |
| **Use cases** | Anomaly detection, disentanglement, compression | Photo-realistic image synthesis |

**In practice:** Hybrid models (VQGAN, Stable Diffusion's VAE encoder) combine the structured latent space of VAEs with the high-quality generation of GAN-like discriminative losses or diffusion models.

---

## Architecture Selection Guide

Choosing the right architecture for a task:

| Data Type | Task | Recommended Architecture |
|---|---|---|
| Images | Classification | ResNet, EfficientNet, ViT |
| Images | Detection / Segmentation | YOLO, Mask R-CNN, SegFormer |
| Images | Generation | Stable Diffusion (DDPM + VAE), GAN |
| Text sequences | Classification | LSTM, BERT, DistilBERT |
| Text sequences | Generation | LSTM (small scale), GPT-style transformer |
| Time series | Forecasting | LSTM, GRU, Temporal Fusion Transformer |
| Tabular | Classification / Regression | MLP, gradient boosting (often beats DL) |
| Audio | Classification | CNN on spectrogram, wav2vec2 |
| Graphs | Node / edge tasks | Graph Neural Network (GNN) |

As a general rule: **start with the simplest architecture that fits your data modality, then scale up complexity only when a simpler model is insufficient.** Pre-trained models (via Hugging Face or torchvision) are almost always a better starting point than training from scratch.

---

## Further Reading

- **CNNs**
  - LeCun, Y. et al. (1998). *Gradient-Based Learning Applied to Document Recognition.* Proceedings of the IEEE.
  - He, K. et al. (2015). *Deep Residual Learning for Image Recognition.* CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
  - Simonyan, K. & Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG).* ICLR 2015. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)

- **RNNs and LSTMs**
  - Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
  - Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder (GRU introduction).* [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)
  - Andrej Karpathy. *The Unreasonable Effectiveness of Recurrent Neural Networks.* [karpathy.github.io/2015/05/21/rnn-effectiveness](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

- **GANs**
  - Goodfellow, I. et al. (2014). *Generative Adversarial Nets.* NeurIPS 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
  - Radford, A. et al. (2015). *Unsupervised Representation Learning with Deep Convolutional GANs (DCGAN).* [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
  - Arjovsky, M. et al. (2017). *Wasserstein GAN.* [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)

- **VAEs**
  - Kingma, D.P. & Welling, M. (2013). *Auto-Encoding Variational Bayes.* ICLR 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
  - Higgins, I. et al. (2017). *beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR 2017.

- **Practical resources**
  - CS231n: Convolutional Neural Networks for Visual Recognition — [cs231n.github.io](https://cs231n.github.io)
  - d2l.ai — Dive into Deep Learning (interactive, multi-framework textbook) — [d2l.ai](https://d2l.ai)
  - PyTorch tutorials — [pytorch.org/tutorials](https://pytorch.org/tutorials)

---

*Last updated: February 2026*
