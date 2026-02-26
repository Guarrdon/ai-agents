# Advanced AI Training Techniques

> **Learning Objectives**
>
> By the end of this document, you will be able to:
> - Explain transfer learning and apply it using pre-trained models in PyTorch
> - Implement fine-tuning strategies including full fine-tuning and parameter-efficient approaches (LoRA, adapters)
> - Describe the full Reinforcement Learning from Human Feedback (RLHF) pipeline and its role in aligning language models
> - Distinguish between RLHF, RLAIF, and Direct Preference Optimization (DPO)
> - Apply multi-task learning and curriculum learning to improve model generalization
> - Recognize practical training challenges (catastrophic forgetting, overfitting) and apply mitigation strategies

---

## 1. Transfer Learning

### 1.1 The Core Idea

Transfer learning exploits the fact that representations learned on one task often generalize to related tasks. Rather than training from a random initialization, you begin with a model that has already developed useful internal representations — edges and textures for vision models, syntactic patterns and semantic relationships for language models — and adapt those representations to your target domain.

**Why it works:** Deep neural networks learn hierarchical representations. Early layers in a vision model learn low-level features (edges, color gradients) that are task-agnostic; later layers learn high-level, task-specific features. When you transfer a model, you keep the reusable low-level features and replace or adapt the task-specific top layers.

### 1.2 Transfer Learning Taxonomy

| Strategy | Description | When to Use |
|---|---|---|
| **Feature extraction** | Freeze all pre-trained layers; train only the new head | Very small datasets; target task similar to source |
| **Fine-tuning (partial)** | Unfreeze top N layers; train head + those layers | Moderate dataset; target task somewhat different |
| **Fine-tuning (full)** | Unfreeze all layers; train end-to-end | Large dataset; significant domain shift |
| **Domain-adaptive pre-training** | Continue pre-training on unlabeled domain data before fine-tuning | Significant vocabulary/distribution gap |

### 1.3 Feature Extraction Example (PyTorch)

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Load pre-trained ResNet-50
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze all parameters
for param in backbone.parameters():
    param.requires_grad = False

# Replace the classification head for a 10-class task
num_features = backbone.fc.in_features
backbone.fc = nn.Linear(num_features, 10)

# Only backbone.fc parameters will receive gradients
trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in backbone.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.1%})")
# Trainable: 20,490 / 25,557,032 (0.1%)
```

### 1.4 Progressive Unfreezing

A practical fine-tuning strategy: unfreeze layers progressively from the top of the network downward, allowing the model to adapt without destroying early representations.

```python
def unfreeze_top_n_layers(model: nn.Module, n: int) -> None:
    """Unfreeze the last n layer groups in a ResNet backbone."""
    layer_groups = [model.layer4, model.layer3, model.layer2, model.layer1, model.conv1]
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    # Then unfreeze the head and top n groups
    for param in model.fc.parameters():
        param.requires_grad = True
    for group in layer_groups[:n]:
        for param in group.parameters():
            param.requires_grad = True

# Epoch 1-5: train head only
# Epoch 6-10: unfreeze layer4
unfreeze_top_n_layers(backbone, n=1)
# Epoch 11+: unfreeze layer4 + layer3
unfreeze_top_n_layers(backbone, n=2)
```

**Learning rate tip:** Use discriminative learning rates — lower learning rates for earlier (more general) layers, higher rates for later (more task-specific) layers:

```python
optimizer = torch.optim.AdamW([
    {"params": backbone.layer1.parameters(), "lr": 1e-5},
    {"params": backbone.layer2.parameters(), "lr": 2e-5},
    {"params": backbone.layer3.parameters(), "lr": 5e-5},
    {"params": backbone.layer4.parameters(), "lr": 1e-4},
    {"params": backbone.fc.parameters(),     "lr": 1e-3},
])
```

### 1.5 Catastrophic Forgetting

A key risk in fine-tuning: the model overrides pre-trained knowledge, degrading performance on the original task.

**Mitigation strategies:**
- **Elastic Weight Consolidation (EWC):** Adds a regularization term that penalizes changes to weights important for the original task.

  `L_total = L_task + λ Σ_i F_i (θ_i - θ*_i)²`

  where `F_i` is the Fisher information (importance weight) for parameter `θ_i` and `θ*_i` is the pre-trained value.

- **Learning Rate Warm-up:** Gradually increase the learning rate from near-zero in early epochs to avoid large destructive updates.
- **Early Stopping:** Monitor validation performance on a representative held-out set; stop training when performance plateaus.

---

## 2. Fine-Tuning Strategies

### 2.1 Full Fine-Tuning

All model parameters are updated. Optimal when you have sufficient labeled data and compute. For large models (billions of parameters), this is prohibitively expensive — motivating parameter-efficient methods.

### 2.2 Parameter-Efficient Fine-Tuning (PEFT)

PEFT methods adapt a large pre-trained model by modifying or adding only a small fraction of parameters, dramatically reducing memory and compute requirements.

#### Adapter Layers

Adapters insert small bottleneck modules after each transformer layer. Only adapter parameters are trained; the original weights remain frozen.

```
[Pre-trained Layer] → [Down-projection: d_model → r] → [Nonlinearity] → [Up-projection: r → d_model] + residual
```

Typical bottleneck dimension `r` is 16–64 (vs. `d_model` of 768–4096), yielding ~0.5–4% of original parameters.

#### Low-Rank Adaptation (LoRA)

LoRA (Hu et al., 2021) decomposes weight updates as the product of two low-rank matrices, rather than adding intermediate layers.

For a weight matrix `W ∈ ℝ^(d×k)`, the update is:

```
W' = W + ΔW = W + BA
```

where `B ∈ ℝ^(d×r)`, `A ∈ ℝ^(r×k)`, and `r << min(d, k)`.

`A` is initialized with a random Gaussian; `B` is initialized to zero, so `ΔW = 0` at the start of training.

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False  # Freeze original weights

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen path + LoRA path
        return (x @ self.weight.T) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
```

At inference time, the LoRA weights can be **merged** back into the base weights (`W' = W + BA * scaling`) with zero added latency.

#### QLoRA

QLoRA (Dettmers et al., 2023) extends LoRA by quantizing the base model to 4-bit precision (NF4 format) during training, reducing memory by ~4×. The LoRA adapters themselves remain in full precision (bfloat16). This enabled fine-tuning of 65B parameter models on a single 48GB GPU.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 6,553,600 || all params: 8,037,269,504 || trainable%: 0.0816
```

#### Prompt Tuning and Prefix Tuning

Instead of modifying model weights, these methods prepend learnable "soft tokens" (continuous embeddings) to the input sequence. The model backbone is fully frozen.

- **Prompt Tuning:** Prepends `k` trainable tokens to the input embeddings only.
- **Prefix Tuning:** Prepends trainable key-value pairs to every attention layer, giving the model more capacity to encode task context.

### 2.3 Instruction Fine-Tuning (SFT)

Supervised Fine-Tuning (SFT) on instruction-following datasets transforms a base language model into a model that reliably follows natural-language instructions.

**Data format:**
```json
{
  "instruction": "Summarize the following article in two sentences.",
  "input": "The James Webb Space Telescope...",
  "output": "The JWST detected... The findings suggest..."
}
```

**Loss:** Standard cross-entropy over the output tokens only (the instruction and input tokens are masked out of the loss computation):

```python
def compute_sft_loss(logits, labels, ignore_index=-100):
    """labels has -100 for instruction tokens, real token IDs for output tokens."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index
    )
    return loss
```

**Data quality > quantity:** A key empirical finding (LIMA, 2023) is that 1,000 high-quality, diverse instruction examples can produce models competitive with those trained on hundreds of thousands of lower-quality examples.

---

## 3. Reinforcement Learning from Human Feedback (RLHF)

RLHF is the dominant technique for aligning large language models to human preferences. It was central to the development of InstructGPT (OpenAI, 2022) and has been adopted in almost every frontier LLM.

### 3.1 The Three-Stage RLHF Pipeline

```
Stage 1: Supervised Fine-Tuning (SFT)
         Base LM → SFT Model (trained on demonstration data)

Stage 2: Reward Model Training
         SFT Model → generates response pairs → human raters rank pairs
         → Train Reward Model (RM) to predict human preference scores

Stage 3: RL Policy Optimization
         SFT Model (init) + Reward Model + KL penalty
         → Optimize policy with PPO to maximize reward
```

### 3.2 Stage 1: Supervised Fine-Tuning

As described in Section 2.3, a base language model is fine-tuned on high-quality human-written demonstrations of desired behavior. This produces the SFT model, which serves as the starting policy for the RL stage and the generator for preference data.

### 3.3 Stage 2: Reward Model Training

**Data collection:** The SFT model generates multiple completions for the same prompt. Human annotators rank these completions by preference (quality, helpfulness, harmlessness).

**Reward model architecture:** Typically the SFT model with its final language modeling head replaced by a scalar regression head:

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.transformer = base_model.transformer  # Frozen or lightly trained
        d_model = base_model.config.hidden_size
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        # Use representation at the last non-padding token
        last_hidden = outputs.last_hidden_state
        # Gather the last actual token's representation
        seq_lengths = attention_mask.sum(dim=1) - 1
        last_token_repr = last_hidden[torch.arange(len(seq_lengths)), seq_lengths]
        reward = self.reward_head(last_token_repr).squeeze(-1)
        return reward
```

**Training loss (Bradley-Terry preference model):**

For a prompt `x` with preferred completion `y_w` (winner) over `y_l` (loser):

```
L_RM = -E[log σ(r(x, y_w) - r(x, y_l))]
```

where `r(x, y)` is the reward model's scalar output and `σ` is the sigmoid function.

```python
def reward_model_loss(reward_chosen, reward_rejected):
    """Bradley-Terry pairwise ranking loss."""
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
```

### 3.4 Stage 3: Policy Optimization with PPO

The SFT model is used as the initial policy `π_θ`. The policy is optimized to maximize the reward model's score, subject to a KL divergence penalty that prevents the policy from drifting too far from the SFT model (which acts as the reference policy `π_ref`).

**Objective:**

```
max_θ E_x~D, y~π_θ [r(x,y)] - β · KL[π_θ(·|x) || π_ref(·|x)]
```

The KL penalty term is crucial: without it, the policy quickly learns to exploit reward model flaws (reward hacking) by producing grammatically plausible but semantically empty outputs that score highly.

**PPO clip objective:**

```
L_CLIP(θ) = E_t [min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t)]
```

where `r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)` is the importance ratio and `Â_t` is the advantage estimate.

```python
def ppo_clip_loss(log_probs_new, log_probs_old, advantages, clip_eps=0.2):
    """PPO clipped surrogate loss."""
    ratio = torch.exp(log_probs_new - log_probs_old)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss

def rlhf_total_loss(reward_score, log_probs_new, log_probs_ref, advantages,
                     kl_coeff=0.02, clip_eps=0.2):
    """Full RLHF training objective."""
    # KL penalty from reference (SFT) model
    kl_penalty = (log_probs_new - log_probs_ref).mean()
    # PPO policy gradient loss
    policy_loss = ppo_clip_loss(log_probs_new, log_probs_new.detach(), advantages, clip_eps)
    # Total: maximize reward - penalize KL drift
    return policy_loss - reward_score.mean() + kl_coeff * kl_penalty
```

### 3.5 Known Challenges in RLHF

| Challenge | Description | Mitigation |
|---|---|---|
| **Reward hacking** | Policy exploits reward model weaknesses | KL penalty; ensemble reward models; iterative RM retraining |
| **Annotator inconsistency** | Human preferences are noisy and subjective | Careful annotator guidelines; inter-annotator agreement thresholds |
| **Mode collapse** | Policy converges to a narrow output distribution | Entropy bonus; temperature during sampling |
| **Scalable oversight** | Humans cannot evaluate superhuman outputs | AI-assisted evaluation (RLAIF); debate protocols |
| **Distribution shift** | RM trained on SFT outputs; policy diverges from that distribution | Online RLHF: continuously collect new preference data |

### 3.6 Alternatives to PPO-based RLHF

#### Reinforcement Learning from AI Feedback (RLAIF)

Instead of human annotators, a large "constitutional" AI model generates preference labels. This dramatically scales the feedback collection pipeline. Claude (Anthropic) uses Constitutional AI (CAI), where a model critiques its own outputs against a set of principles and self-revises.

#### Direct Preference Optimization (DPO)

DPO (Rafailov et al., 2023) eliminates the separate reward model training stage by deriving a closed-form relationship between the optimal reward function and the optimal policy. This allows training directly on preference pairs using a classification loss:

```
L_DPO(θ) = -E[(y_w, y_l)~D] [log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

```python
def dpo_loss(policy_log_probs_chosen, policy_log_probs_rejected,
             ref_log_probs_chosen, ref_log_probs_rejected, beta=0.1):
    """Direct Preference Optimization loss."""
    log_ratio_chosen = policy_log_probs_chosen - ref_log_probs_chosen
    log_ratio_rejected = policy_log_probs_rejected - ref_log_probs_rejected
    loss = -torch.log(torch.sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))).mean()
    return loss
```

**DPO advantages over PPO-RLHF:**
- No separate reward model to train and maintain
- No online sampling loop (simpler training pipeline)
- More stable optimization

**DPO limitations:**
- Fixed reference policy may not be optimal for all tasks
- Less flexibility for iterative data collection
- Can underfit with insufficient preference data

---

## 4. Multi-Task and Curriculum Learning

### 4.1 Multi-Task Learning

Training a single model jointly on multiple related tasks typically improves generalization by forcing the model to learn shared representations. The loss is a weighted combination of per-task losses:

```
L_total = Σ_i w_i · L_i
```

**Weight selection strategies:**
- **Fixed weights:** Manually tuned, often equal
- **Uncertainty weighting (Kendall et al., 2018):** `w_i = 1 / (2σ_i²)` where `σ_i` is a learnable task-specific uncertainty parameter
- **GradNorm:** Dynamically adjusts weights so gradient norms are balanced across tasks

```python
class MultiTaskModel(nn.Module):
    def __init__(self, backbone, num_tasks):
        super().__init__()
        self.backbone = backbone
        self.task_heads = nn.ModuleList([
            nn.Linear(backbone.hidden_size, task_output_size)
            for task_output_size in num_tasks
        ])
        # Learnable log-variance for uncertainty weighting
        self.log_vars = nn.Parameter(torch.zeros(len(num_tasks)))

    def forward(self, x, task_id):
        features = self.backbone(x)
        return self.task_heads[task_id](features)

    def uncertainty_weighted_loss(self, losses):
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]
        return total
```

### 4.2 Curriculum Learning

Training on examples in a structured order from easy to hard (Bengio et al., 2009). Rather than sampling uniformly from the training set, curriculum learning uses a difficulty measure to sequence examples.

**Difficulty metrics:**
- Loss value on a well-trained reference model (high loss = harder)
- Length/complexity of the example
- Number of training steps required for a small model to converge on that example

**Self-paced learning:** The model itself determines example difficulty dynamically. Examples where the current model's loss exceeds a threshold `λ` are considered too hard and temporarily excluded. `λ` is gradually increased as training progresses.

```python
def self_paced_weights(losses: torch.Tensor, threshold: float) -> torch.Tensor:
    """Binary weights: 1 if loss < threshold (easy), 0 otherwise (too hard)."""
    return (losses < threshold).float()

def self_paced_loss(model_losses: torch.Tensor, threshold: float) -> torch.Tensor:
    weights = self_paced_weights(model_losses, threshold)
    return (weights * model_losses).sum() / (weights.sum() + 1e-8)
```

---

## 5. Practical Training Considerations

### 5.1 Data Augmentation for Fine-Tuning

When labeled data is scarce, augmentation artificially expands the training set.

**For NLP:**
- **Back-translation:** Translate to another language and back; preserves semantics while varying surface form
- **Synonym substitution:** Replace non-stopwords with WordNet synonyms
- **EDA (Wei & Zou, 2019):** Synonym replacement, random insertion, random swap, random deletion

**For vision:**
- Random cropping, horizontal flipping, color jitter (standard)
- **MixUp:** Linear interpolation of two training examples and their labels
- **CutMix:** Paste patches from one image onto another
- **RandAugment:** Automatically search over augmentation policies

### 5.2 Learning Rate Scheduling

| Schedule | Formula | Use Case |
|---|---|---|
| **Step decay** | `lr × γ^(epoch/step_size)` | Classic; simple to tune |
| **Cosine annealing** | `lr_min + 0.5(lr_max - lr_min)(1 + cos(πt/T))` | LLM fine-tuning; smooth decay |
| **Warm-up + cosine** | Linear warm-up then cosine decay | Transformer training; prevents early instability |
| **One-cycle** | Rise then fall within a single cycle | Fast convergence; super-convergence |

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                                  total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps,
                                          eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                        milestones=[warmup_steps])
```

### 5.3 Mixed-Precision Training

Training with 16-bit floats (float16 or bfloat16) reduces memory by ~2× and increases throughput on modern GPUs with Tensor Cores.

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    with autocast(dtype=torch.bfloat16):  # bfloat16 preferred for training stability
        outputs = model(batch["input_ids"])
        loss = criterion(outputs, batch["labels"])

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

### 5.4 Gradient Checkpointing

Trades compute for memory: instead of storing all intermediate activations during the forward pass, recompute them during backpropagation. Reduces activation memory from `O(L)` to `O(√L)` for an `L`-layer network.

```python
from torch.utils.checkpoint import checkpoint_sequential

# Instead of:
output = model(input)

# Use (approximately):
output = checkpoint_sequential(model.layers, segments=4, input=input)
```

---

## Further Reading

- **LoRA** — Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **InstructGPT/RLHF** — Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback.* [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- **DPO** — Rafailov, R. et al. (2023). *Direct Preference Optimization.* [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- **QLoRA** — Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **Constitutional AI** — Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
- **LIMA** — Zhou, C. et al. (2023). *LIMA: Less Is More for Alignment.* [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)
- **EWC** — Kirkpatrick, J. et al. (2017). *Overcoming catastrophic forgetting in neural networks.* [PNAS](https://www.pnas.org/doi/10.1073/pnas.1611835114)
- **Hugging Face PEFT Library** — [github.com/huggingface/peft](https://github.com/huggingface/peft)
- **TRL (Transformer Reinforcement Learning)** — [github.com/huggingface/trl](https://github.com/huggingface/trl)

---

*Last updated: February 2026*
