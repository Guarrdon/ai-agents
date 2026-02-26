# Model Architectures

> 📍 [Home](../../README.md) › [Advanced](../README.md) › Model Architectures

---

## Learning Objectives

By the end of this document you will be able to:

- Explain the Transformer architecture and each of its components
- Describe how self-attention computes relevance between tokens
- Compare multi-head attention variants (MHA, MQA, GQA, FlashAttention)
- Understand position encoding schemes (absolute, RoPE, ALiBi)
- Distinguish Mixture-of-Experts (MoE) from dense models
- Summarise state space models (SSMs/Mamba) and their advantages

---

## Prerequisites

| Topic | Where to Review |
|-------|----------------|
| Matrix multiplication and dot products | Linear algebra fundamentals |
| Softmax function | Neural Networks ([neural-networks.md](neural-networks.md)) |
| Feedforward networks | [Neural Networks](neural-networks.md) |

---

## 1. The Transformer Architecture

Introduced in *"Attention Is All You Need"* (Vaswani et al., 2017), the Transformer replaced recurrence with **self-attention**, enabling massive parallelism and long-range dependency modelling.

### High-Level Structure

```
Input Tokens
     ↓
Token Embeddings + Position Encoding
     ↓
┌─────────────────────────────┐
│   Transformer Block × N     │
│                              │
│  ┌──────────────────────┐   │
│  │  Multi-Head Attention │   │
│  └──────────────────────┘   │
│           ↓ + residual       │
│    Layer Normalisation       │
│           ↓                  │
│  ┌──────────────────────┐   │
│  │  Feed-Forward Network│   │
│  └──────────────────────┘   │
│           ↓ + residual       │
│    Layer Normalisation       │
└─────────────────────────────┘
     ↓
 Output Head (language model head, classifier, etc.)
```

---

## 2. Self-Attention

Self-attention allows each position in a sequence to attend to all other positions, computing a weighted sum based on relevance.

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

Where:
- **Q** (Queries): "What am I looking for?"
- **K** (Keys): "What do I contain?"
- **V** (Values): "What do I output if selected?"
- **d_k**: Key dimension (scaling prevents gradient vanishing in softmax)

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    q: torch.Tensor,  # (batch, heads, seq_len, d_k)
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)
```

---

## 3. Multi-Head Attention Variants

Running attention in multiple parallel "heads" allows the model to attend to different aspects of the sequence simultaneously.

### Multi-Head Attention (MHA)

Each head has its own Q, K, V projections — `h` heads × `d_model/h` dimension each.

**Memory cost:** `O(n² · d_model)` where `n` is sequence length — expensive for long sequences.

### Multi-Query Attention (MQA)

All heads share a single K and V projection. Reduces memory bandwidth dramatically during inference.

**Trade-off:** Slightly lower quality than MHA; used in models like PaLM and Falcon.

### Grouped Query Attention (GQA)

`g` groups of heads share K/V projections (g < num_heads). Balances quality vs memory efficiency.

**Used in:** LLaMA 2/3, Mistral, Gemma — the current practical standard.

| Variant | KV heads | Quality | Inference memory |
|---------|---------|---------|-----------------|
| MHA | = Q heads | Best | Highest |
| GQA | `g` groups | Near-MHA | Medium |
| MQA | 1 | Slightly lower | Lowest |

### FlashAttention

An **algorithm** (not a new architecture) that reorders attention computation to exploit GPU memory hierarchy:

- Tiles Q, K, V to keep data in fast SRAM
- Avoids materialising the full `n × n` attention matrix
- **Exact same output as standard attention** — no approximation
- 2–4× faster, O(n) memory instead of O(n²)

```python
# PyTorch 2.0+ includes FlashAttention via scaled_dot_product_attention
output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

---

## 4. Position Encodings

Transformers have no inherent notion of position — position encodings inject this information.

### Absolute Sinusoidal (original Transformer)

Fixed, deterministic encodings added to embeddings. Works but doesn't generalise well to lengths beyond training.

### Rotary Position Embedding (RoPE)

Encodes relative positions by rotating Q and K vectors. Token at position `m` has its Q/K rotated by angle `mθ`:

```
Benefits:
✓ Naturally encodes relative positions
✓ Extrapolates to longer contexts with extensions (YaRN, LongRoPE)
✓ Used in: LLaMA, Mistral, Gemma, Qwen, Falcon
```

### ALiBi (Attention with Linear Biases)

Adds a linear penalty to attention scores based on distance between tokens:

```
score(i, j) = qᵢ · kⱼ / √d_k  - m|i - j|
```

Where `m` is a head-specific slope. Extrapolates to longer sequences naturally.

---

## 5. Feed-Forward Sublayer

Each Transformer block contains a two-layer MLP applied position-wise:

```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Standard: d_ff = 4 × d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # gate projection (SwiGLU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU variant (used in LLaMA)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

**SwiGLU** (SiLU-gated linear unit) outperforms the original ReLU-based FFN and is now standard in most open-weight LLMs.

---

## 6. Normalisation

### Pre-Norm vs Post-Norm

Original Transformer used **Post-Norm** (LayerNorm after residual connection). Modern LLMs use **Pre-Norm** (LayerNorm before each sublayer) for more stable training of deep models.

### RMSNorm

A simplified form of LayerNorm that only rescales (no recentering):

```
RMSNorm(x) = x / RMS(x) · γ
```

Cheaper to compute, equally effective — used in LLaMA, Mistral, Gemma.

---

## 7. Mixture of Experts (MoE)

MoE replaces the dense FFN with `N` expert FFNs and a routing mechanism that activates only `k` of them per token.

```
Sparse MoE FFN:
  router scores = Softmax(W_r · x)
  top-k experts selected
  output = weighted sum of top-k expert outputs
```

**Key benefit:** Scales parameter count without proportionally scaling compute.

| | Dense | MoE |
|-|-------|-----|
| Parameters | d | N × d |
| Active params per token | d | k × d/N |
| Training FLOPs | ∝ d | ∝ k × d/N |
| Examples | LLaMA, Mistral | Mixtral 8×7B, GPT-4 (rumoured), DeepSeek |

**Challenges:** Load balancing across experts, communication overhead in distributed settings.

---

## 8. State Space Models (SSMs)

SSMs (e.g., **Mamba**) offer an alternative to attention for long-sequence modelling.

### Core Idea

SSMs model sequences as a linear recurrence:

```
h_t = A · h_{t-1} + B · x_t     (hidden state update)
y_t = C · h_t                    (output)
```

Where A, B, C are learnable parameters.

### Mamba

Mamba introduces **selective state spaces** — the A, B, C matrices become input-dependent, giving the model the ability to selectively remember or forget information.

| | Transformer | Mamba |
|-|-------------|-------|
| Training | Parallel | Parallel (parallel scan) |
| Inference | KV cache grows with context | Fixed-size state |
| Attention complexity | O(n²) | O(n) |
| Long-context scaling | Expensive | Efficient |
| Short-context quality | Excellent | Competitive |

**Hybrid models** (e.g., Jamba, Zamba) combine Transformer and Mamba layers to get the best of both.

---

## 9. Scaling Laws

Scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022 — "Chinchilla") describe how loss improves as a power law with compute, parameters, and data.

### Chinchilla Optimal Scaling

For a compute budget `C`, the optimal allocation is:

```
N_opt ∝ C^0.5    (model parameters)
D_opt ∝ C^0.5    (training tokens)
```

Rule of thumb: train a model for **~20 tokens per parameter** for compute-optimal training.

> **Example:** A 7B parameter model is compute-optimally trained on ~140B tokens.

---

## Key Concepts Summary

| Concept | One-line description |
|---------|---------------------|
| **Self-attention** | Each token attends to all others; computes weighted sum of values |
| **Multi-head attention** | Multiple parallel attention heads capturing different aspects |
| **GQA** | Grouped Query Attention — shared KV heads for memory efficiency |
| **FlashAttention** | Exact attention with O(n) memory via tiling on GPU SRAM |
| **RoPE** | Rotary position embeddings encoding relative positions |
| **MoE** | Sparse mixture of experts — more parameters, same compute |
| **SSM/Mamba** | Recurrence-based alternative to attention; O(n) inference |
| **Scaling laws** | Loss improves predictably with compute, params, data |

---

## Further Reading

| Resource | Type | Notes |
|---------|------|-------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Paper | Original Transformer paper |
| [Flash Attention 2](https://arxiv.org/abs/2307.08691) | Paper | Memory-efficient exact attention |
| [GQA: Training Generalised Multi-Query Attention](https://arxiv.org/abs/2305.13245) | Paper | GQA derivation and results |
| [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752) | Paper | Selective state space models |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | Paper | Original OpenAI scaling laws |
| [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) | Paper | Revised optimal scaling |
| [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) | Blog | Excellent visual walkthrough |

---

*Navigation: [← Neural Networks](neural-networks.md) · [Advanced Home](../README.md) · [Next: Large Language Models →](large-language-models.md)*
