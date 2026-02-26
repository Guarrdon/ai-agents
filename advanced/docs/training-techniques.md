# Training Techniques

> 📍 [Home](../../README.md) › [Advanced](../README.md) › Training Techniques

---

## Learning Objectives

By the end of this document you will be able to:

- Apply transfer learning strategies including feature extraction and progressive unfreezing
- Compare modern optimisers (AdamW, Lion, Muon) and their trade-offs
- Design a learning rate schedule appropriate for LLM pre-training
- Explain the different parallelism strategies for distributed training
- Configure mixed-precision training (BF16/FP8) safely
- Debug common training instabilities such as loss spikes

---

## Prerequisites

| Topic | Where to Review |
|-------|----------------|
| Gradient descent and backpropagation | [Neural Networks](neural-networks.md) |
| Transformer architecture | [Model Architectures](model-architectures.md) |
| PyTorch basics | PyTorch official tutorials |

---

## 1. Transfer Learning

Transfer learning exploits the fact that representations learned on one task often generalise to related tasks. Rather than training from a random initialisation, you begin with a model that has already developed useful internal representations and adapt them to your target domain.

### Feature Extraction vs Fine-Tuning

| Strategy | Description | When to Use |
|---|---|---|
| **Feature extraction** | Freeze all pre-trained layers; train only the new head | Very small datasets; target task similar to source |
| **Fine-tuning (partial)** | Unfreeze top N layers; train head + those layers | Moderate dataset; target task somewhat different |
| **Fine-tuning (full)** | Unfreeze all layers; train end-to-end | Large dataset; significant domain shift |

### Feature Extraction Example (PyTorch)

```python
import torchvision.models as models
import torch.nn as nn

# Load pre-trained ResNet-50
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze all parameters
for param in backbone.parameters():
    param.requires_grad = False

# Replace the classification head for a 10-class task
num_features = backbone.fc.in_features
backbone.fc = nn.Linear(num_features, 10)
# Only backbone.fc parameters receive gradients
```

### Progressive Unfreezing

Unfreeze layers progressively from the top downward, allowing the model to adapt without destroying early representations:

```python
def unfreeze_top_n_layers(model: nn.Module, n: int) -> None:
    layer_groups = [model.layer4, model.layer3, model.layer2, model.layer1]
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    for group in layer_groups[:n]:
        for param in group.parameters():
            param.requires_grad = True
```

**Discriminative learning rates:** Use lower LRs for earlier (general) layers, higher for later (task-specific) layers.

### Catastrophic Forgetting

Fine-tuning can override pre-trained knowledge. Mitigations:
- **Elastic Weight Consolidation (EWC):** Penalises changes to weights important for the original task: `L_total = L_task + λ Σ F_i (θ_i - θ*_i)²`
- **Learning Rate Warm-up:** Gradually increase LR from near-zero to avoid large destructive updates
- **Early Stopping:** Monitor validation performance and stop when it plateaus

---

## 2. Optimisers

### AdamW

The standard choice for LLM training. AdamW decouples weight decay from the gradient update, unlike the original Adam which incorrectly absorbed it into the adaptive learning rate.

```python
import torch

optimiser = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),   # β₁=0.9 (momentum), β₂=0.95 (variance)
    weight_decay=0.1,    # decoupled L2 regularisation
    eps=1e-8,
)
```

**Memory cost:** 2 extra state tensors per parameter (first and second moment) = 12 bytes/param in FP32.

### Lion

Lion (EvoLved Sign Momentum) uses only the sign of the gradient update:

```
update = sign(β₁ · m_{t-1} + (1 - β₁) · g_t)
m_t = β₂ · m_{t-1} + (1 - β₂) · g_t
θ_t = θ_{t-1} - η · (update + λθ_{t-1})
```

**Advantages:** 1 state tensor per parameter (vs 2 for Adam) = 33% memory saving.
**Trade-off:** Often requires lower learning rates; behaviour differs from AdamW.

### Muon

Muon (MomentUm Orthogonalised by Newton-schulz) orthogonalises gradient updates for 2D parameter matrices using the Nesterov momentum update. Reported strong results on language model pre-training. Still actively researched.

---

## 3. Learning Rate Schedules

### Warmup + Cosine Decay

The standard schedule for LLM pre-training:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

# Linear warmup for first N steps, then cosine decay
def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

### WSD (Warmup-Stable-Decay)

An alternative to cosine: hold LR constant after warmup (stable phase), then decay sharply at the end. Used in MiniCPM, DeepSeek. Allows extending training without restarting the schedule.

```
Phase 1: Linear warmup
Phase 2: Constant LR (majority of training)
Phase 3: Sharp cosine/linear decay
```

---

## 4. Data Pipeline

### Sequence Packing

Avoids padding by concatenating multiple short documents into a single long sequence:

```
Without packing: [doc_1, PAD, PAD, PAD | doc_2, PAD, PAD | ...]
With packing:    [doc_1, doc_2, doc_3, doc_4_part | doc_4_cont, ...]
```

Adds special attention masking to prevent cross-document attention.

### Data Deduplication

Duplicate data causes memorisation and reduces effective diversity. Standard pipeline:
1. MinHash LSH near-deduplication (within and across sources)
2. Exact substring deduplication
3. Quality filtering (perplexity scoring, classifier-based)

### Data Mixture and Curriculum

Mix data from different domains (web, code, books, math) with carefully tuned weights. Research shows benefits from:
- Starting with higher-quality data early in training
- Gradually increasing difficulty
- Upsampling rare but high-quality sources

---

## 5. Distributed Training

Modern LLMs exceed the memory of a single GPU. Distributed training spreads computation across many devices.

### Data Parallelism (DP)

Each GPU holds a complete model copy; processes different mini-batches; gradients are averaged across GPUs.

```
GPU 0: model copy + batch slice A → gradients
GPU 1: model copy + batch slice B → gradients
GPU 2: model copy + batch slice C → gradients
...
AllReduce: average gradients, update all copies identically
```

**Limit:** Model must fit in a single GPU.

### Fully Sharded Data Parallel (FSDP)

Shards model parameters, gradients, and optimiser states across GPUs. Each GPU holds a **shard**; all-gather before each forward/backward computation.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
)
```

PyTorch 2.x `FSDP2` (DTensor-based) is the current recommended approach.

### Tensor Parallelism (TP)

Shards individual weight matrices across GPUs. Requires changing the model implementation; used for very large models where FSDP alone is insufficient.

### Pipeline Parallelism (PP)

Splits model layers across GPUs in a pipeline; different GPUs process different micro-batches simultaneously.

### 3D Parallelism

Combines DP + TP + PP for the largest models (GPT-4, Llama-3 405B). Each dimension handles a different aspect of scale.

---

## 6. Mixed Precision Training

Training in lower precision reduces memory and increases throughput.

### BF16 (Brain Float 16)

The standard precision for modern LLM training:

```
FP32: 1 sign, 8 exponent, 23 mantissa bits — full precision
BF16: 1 sign, 8 exponent,  7 mantissa bits — same range as FP32, less precision
FP16: 1 sign, 5 exponent, 10 mantissa bits — smaller range, overflow risk
```

BF16 is preferred over FP16 because it shares FP32's dynamic range, avoiding loss spikes from overflow.

```python
# Automatic mixed precision with BF16
from torch.amp import autocast

with autocast(device_type='cuda', dtype=torch.bfloat16):
    loss = model(x, labels=y)

loss.backward()
```

### FP8 Training

FP8 (e.g., on H100 GPUs) halves memory further. Requires:
- Scaling factors per-tensor or per-block
- Careful handling of gradient accumulation
- Supported via `transformer_engine` or `torchao`

---

## 7. Gradient Checkpointing

**Problem:** Storing all intermediate activations for backprop is memory-intensive.
**Solution:** Recompute activations during the backward pass instead of storing them.

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerBlock(nn.Module):
    def forward(self, x):
        # Recompute activations during backward pass
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        # Normal forward pass
        ...
```

**Trade-off:** ~30–40% training slowdown in exchange for significant memory savings. Often used selectively on only some layers.

---

## 8. Gradient Accumulation

Simulates large batch sizes on limited GPU memory by accumulating gradients over multiple forward/backward passes before stepping:

```python
accumulation_steps = 8  # effective batch = batch_size × accumulation_steps

optimiser.zero_grad()
for step, batch in enumerate(dataloader):
    loss = model(**batch).loss / accumulation_steps
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimiser.step()
        optimiser.zero_grad()
```

---

## 9. Training Stability

### Common Instabilities

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss spike then recovery | Noisy data batch | Gradient clipping; data filtering |
| Loss spike, no recovery | LR too high | Reduce LR; restart from checkpoint |
| NaN loss | Overflow | Switch to BF16; reduce LR |
| Diverging loss | All of the above | Start smaller; systematic search |

### Gradient Clipping

Caps gradient norm to prevent exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Standard values: `max_norm=1.0` for most LLM training.

### μP (maximal update Parametrisation)

μP rescales initialisations and learning rates based on model width, ensuring hyperparameters transfer from small proxy models to large production models — dramatically reducing expensive hyperparameter sweeps.

---

## Key Concepts Summary

| Concept | One-line description |
|---------|---------------------|
| **AdamW** | Adam with decoupled weight decay; standard LLM optimiser |
| **WSD schedule** | Warmup → Stable → Decay; flexible for extending training |
| **FSDP** | Shards model state across GPUs for memory efficiency |
| **BF16** | 16-bit format with FP32's dynamic range; standard training precision |
| **Gradient checkpointing** | Recompute activations on backward pass to save memory |
| **μP** | Width-invariant hyperparameter scaling enabling proxy training |

---

## 🧠 Knowledge Check

Test your understanding. Attempt each question before revealing the answer.

**Q1.** What is the primary purpose of gradient clipping in large model training, and how does it work?

<details>
<summary>Answer</summary>

**Purpose:** Prevent exploding gradients from destabilising training. If a gradient vector is abnormally large (due to a difficult example or numerical instability), unconstrained updates can push weights far outside a useful range, causing a loss spike or training divergence.

**How it works:** Before the optimiser step, the global gradient norm is computed. If it exceeds a threshold (e.g., 1.0), all gradients are scaled down proportionally so the norm equals the threshold:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This preserves gradient direction while bounding step size.

</details>

---

**Q2.** What is the key difference between Tensor Parallelism and Pipeline Parallelism in distributed training?

<details>
<summary>Answer</summary>

- **Tensor Parallelism (TP):** Splits individual weight matrices across devices. For example, an MLP layer's weight matrix is column-split across 8 GPUs, each computing a partial matrix multiplication. Requires AllReduce communication at each layer boundary. Good for very large layers (attention, FFN in transformers).

- **Pipeline Parallelism (PP):** Assigns consecutive layers (a "stage") to each device. Data flows through the pipeline sequentially. Communication is cheaper (only activations at stage boundaries) but requires careful microbatch scheduling to avoid GPU idle time ("pipeline bubbles").

Both are commonly combined with Data Parallelism (DP) as 3D parallelism (TP × PP × DP) for training the largest models.

</details>

---

**Q3.** Why is BF16 preferred over FP16 for LLM training despite having less numerical precision?

<details>
<summary>Answer</summary>

BF16 (Brain Float 16) has 8 exponent bits and 7 mantissa bits, while FP16 has 5 exponent bits and 10 mantissa bits.

- **FP16** has higher precision in its mantissa but a very limited dynamic range (max value ~65,504). Large gradient values overflow to NaN/Inf, requiring loss scaling hacks and careful monitoring.
- **BF16** matches FP32's dynamic range (same 8-bit exponent, same max value), making it numerically stable for the wide range of gradient magnitudes encountered in LLM training — without requiring loss scaling.

The lower mantissa precision of BF16 has negligible impact on final model quality in practice. Most modern LLM training (LLaMA, GPT-4, Gemini) uses BF16 as the default mixed-precision format.

</details>

---

➡️ **Full quiz with 3 questions:** [Knowledge Checks → Training Techniques](knowledge-checks.md#7-training-techniques)

---

## Further Reading

| Resource | Type | Notes |
|---------|------|-------|
| [AdamW paper](https://arxiv.org/abs/1711.05101) | Paper | Decoupled weight decay |
| [Lion optimiser](https://arxiv.org/abs/2302.06675) | Paper | Sign-based momentum optimiser |
| [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | Documentation | Official FSDP guide |
| [Mixed Precision Training](https://arxiv.org/abs/1710.03740) | Paper | Original AMP paper |
| [μP paper](https://arxiv.org/abs/2203.03466) | Paper | Maximal update parametrisation |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Code | Reference implementation of 3D parallelism |

---

*Navigation: [← Prompt Engineering](prompt-engineering.md) · [Advanced Home](../README.md) · [Next: Fine-Tuning →](fine-tuning.md)*
