# Fine-Tuning

> 📍 [Home](../../README.md) › [Advanced](../README.md) › Fine-Tuning

---

## Learning Objectives

By the end of this document you will be able to:

- Distinguish full fine-tuning from parameter-efficient methods
- Implement LoRA and understand its mathematical derivation
- Compare PEFT methods (LoRA, DoRA, QLoRA, GaLore) by use case
- Understand instruction tuning and its data requirements
- Explain DPO and how it differs from RLHF

---

## Prerequisites

| Topic | Where to Review |
|-------|----------------|
| Transformer architecture | [Model Architectures](model-architectures.md) |
| Training loop fundamentals | [Training Techniques](training-techniques.md) |
| What fine-tuning is conceptually | [Beginner: How to Use AI](../../beginner/docs/how-to-use-ai.md) (conceptual introduction) |

---

## 1. Full Fine-Tuning

Full fine-tuning updates all model parameters on a supervised dataset. It achieves the best performance but has significant costs:

```python
# Full fine-tuning: all parameters are trainable
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
trainable_params = sum(p.numel() for p in model.parameters())
# For a 7B model: ~14 GB in BF16, ~56 GB optimiser states in FP32
```

**When to use:**
- You have large, high-quality task-specific data (>100K examples)
- Maximum performance is required
- Compute cost is not a constraint

**Risks:**
- **Catastrophic forgetting:** Overwriting general capabilities when training on narrow data
- **Mitigation:** Regularisation, data mixing, early stopping

---

## 2. Parameter-Efficient Fine-Tuning (PEFT)

PEFT methods freeze most model weights and update only a small fraction, achieving comparable results with a fraction of the memory and compute.

### Low-Rank Adaptation (LoRA)

LoRA decomposes the weight update into two low-rank matrices:

```
W' = W + ΔW = W + BA
```

Where:
- `W ∈ R^{d×k}` is the frozen pre-trained weight
- `B ∈ R^{d×r}` and `A ∈ R^{r×k}` are trainable, with rank `r << min(d, k)`
- Trainable parameters: `r(d + k)` instead of `dk` — e.g., 1% of original

```python
from peft import get_peft_model, LoraConfig, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,              # rank (8, 16, 32, 64 are common)
    lora_alpha=32,     # scaling factor (α/r is the effective LR scale)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06
```

**Choosing rank `r`:**
- `r=8–16`: Good for most instruction-following tasks
- `r=32–64`: More expressive; for complex tasks or larger updates
- Diminishing returns beyond `r=64` in most cases

### QLoRA

Quantise the base model to 4-bit (NF4 quantisation), apply LoRA adapters in BF16. Dramatically reduces memory — enables fine-tuning a 70B model on a single 48GB GPU.

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # double quantisation reduces memory further
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70B",
    quantization_config=bnb_config,
)
model = get_peft_model(model, lora_config)
```

### DoRA (Weight-Decomposed Low-Rank Adaptation)

DoRA decomposes pre-trained weights into **magnitude** and **direction** components, updating both separately. Achieves higher quality than LoRA at the same parameter budget, especially for reasoning tasks.

### GaLore (Gradient Low-Rank Projection)

Projects gradients into a low-rank subspace during full fine-tuning, reducing optimiser state memory without requiring a PEFT architecture. Allows approximate full fine-tuning at PEFT memory costs.

### PEFT Method Comparison

| Method | Memory saving | Quality | Training speed | Notes |
|--------|--------------|---------|----------------|-------|
| Full FT | None | Best | Baseline | Needs high-end hardware |
| LoRA | ~70% | Very good | Fast | Most widely used |
| QLoRA | ~90% | Good | Moderate | Best for consumer hardware |
| DoRA | ~70% | Better than LoRA | Similar to LoRA | Better for complex tasks |
| GaLore | ~60% | Near full FT | Similar to FT | Full weight updates |

---

## 3. Instruction Tuning

Instruction tuning trains a model to follow natural language instructions by fine-tuning on (instruction, response) pairs.

### Data Format

```json
{
  "instruction": "Summarise the following article in 3 bullet points.",
  "input": "...[article text]...",
  "output": "• First key point\n• Second key point\n• Third key point"
}
```

### Chat Template

Modern models use a structured chat template:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Explain quantum entanglement simply.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

### Data Quality Over Quantity

Research consistently shows that a small dataset of high-quality, diverse instructions outperforms large noisy datasets:
- LIMA (2023): 1,000 carefully curated examples achieved near-GPT-4-level instruction following
- **Filtering:** Remove low-quality, harmful, or duplicated examples
- **Diversity:** Cover a wide range of task types

---

## 4. Alignment Fine-Tuning

After instruction tuning, alignment fine-tuning makes models helpful, harmless, and honest.

### RLHF (Reinforcement Learning from Human Feedback)

The original alignment approach (used for InstructGPT, ChatGPT):

```
Step 1: SFT (Supervised Fine-Tuning)
  → Fine-tune on curated instruction-following data

Step 2: Reward Modelling
  → Train a reward model on human preference data (comparisons)
  → Input: (prompt, response_A, response_B); Output: which is better

Step 3: PPO (RL optimisation)
  → Use PPO to optimise policy to maximise reward model score
  → KL penalty to prevent reward hacking (drifting from SFT model)
```

### DPO (Direct Preference Optimisation)

DPO eliminates the separate reward model, directly optimising the policy from preference data:

```python
# DPO loss (simplified)
# Given: prompt, chosen_response, rejected_response
# Reference model (frozen SFT model) provides baseline log-probs

log_ratio = (log_prob(chosen | policy) - log_prob(chosen | reference)) \
          - (log_prob(rejected | policy) - log_prob(rejected | reference))

loss = -F.logsigmoid(β * log_ratio)
```

β controls how strongly the model is pushed toward preferences (typically 0.1–0.5).

### DPO Variants

| Variant | Key Difference | Notes |
|---------|---------------|-------|
| **DPO** | Reference model KL constraint | Original; widely used |
| **IPO** | Regularised with identity function | More stable than DPO |
| **KTO** | Uses individual ratings not pairs | Works with non-paired data |
| **ORPO** | Combines SFT + alignment loss | No reference model needed |
| **SimPO** | Length-normalised; no reference | Simpler, strong results |

---

## 5. Reward Modelling

Reward models (RMs) score responses for quality, used in RLHF and best-of-N sampling.

### Outcome Reward Models (ORMs)

Score complete responses. Common for alignment (helpfulness, harmlessness, honesty).

### Process Reward Models (PRMs)

Score each step in a chain-of-thought reasoning trace. More expensive to label but much better for math/coding tasks where intermediate steps matter.

### Best-of-N Sampling

A simple but effective alternative to RL: generate N responses, score with a reward model, return the highest-scoring response. Competitive with PPO at lower complexity.

---

## 6. Weight Merging

Merging multiple fine-tuned adapters or full models without additional training.

### TIES Merging

Resolves parameter conflicts by:
1. Trimming small parameter changes
2. Electing the sign with the most support
3. Averaging only same-sign parameters

### DARE

Randomly prunes fine-tuned weight deltas (typically 90–99%) before merging, reducing interference between models.

### Model Soup

Simple averaging of model weights from multiple checkpoints or models fine-tuned on different tasks. Improves robustness and generalisation.

---

## Key Concepts Summary

| Concept | One-line description |
|---------|---------------------|
| **LoRA** | Low-rank weight updates — fine-tune with 0.1% of parameters |
| **QLoRA** | LoRA on a 4-bit quantised base model — consumer GPU fine-tuning |
| **Instruction tuning** | Train on (instruction, response) pairs to follow natural language |
| **RLHF** | Three-stage alignment: SFT → reward model → PPO |
| **DPO** | Direct preference optimisation — alignment without a reward model |
| **PRM** | Process reward model — scores reasoning steps, not just final answers |

---

## 🧠 Knowledge Check

Test your understanding. Attempt each question before revealing the answer.

**Q1.** How does LoRA reduce the number of trainable parameters compared to full fine-tuning?

<details>
<summary>Answer</summary>

LoRA (Low-Rank Adaptation) keeps the original pre-trained weights `W₀` **frozen** and adds a small trainable update expressed as a product of two low-rank matrices: `ΔW = A · B`, where A ∈ ℝ^(d×r) and B ∈ ℝ^(r×k), with rank `r << min(d, k)`.

The forward pass uses `W₀ + A · B` without modifying W₀.

**Parameter savings example:** For a weight matrix W with d=4096, k=4096 (16.7M parameters), using rank r=16 requires only `(4096×16) + (16×4096) = 131,072` trainable parameters — less than 1% of the original. For a 7B parameter model, this typically means ~5-20M trainable parameters instead of 7 billion.

</details>

---

**Q2.** What advantage does DPO have over PPO-based RLHF for alignment fine-tuning?

<details>
<summary>Answer</summary>

DPO (Direct Preference Optimisation) eliminates the need for a **separate reward model** and the **RL training loop**:

- **PPO RLHF** requires: (1) training a reward model on preference data, (2) running PPO with 3–4 model copies in memory (policy, reference, reward model, and optionally value model), (3) tuning RL-specific hyperparameters (KL coefficient, clipping range, etc.).

- **DPO** reformulates the RLHF objective into a supervised binary cross-entropy loss directly on (preferred, rejected) response pairs, with the reference model (SFT model) used only for computing a log-probability ratio — no reward model needed.

DPO is simpler to implement, more memory-efficient, more numerically stable, and competitive with PPO on most benchmarks. It has become the dominant alignment method.

</details>

---

**Q3.** What does QLoRA enable, and what technique makes it possible?

<details>
<summary>Answer</summary>

**QLoRA** enables fine-tuning large models (e.g., 65B parameters) on a **single consumer GPU** (e.g., 48GB A6000 or even 24GB RTX 3090) by combining:

1. **4-bit NormalFloat (NF4) quantisation** of the frozen base model weights — reducing memory from ~130 GB (FP32) to ~33 GB (4-bit), while preserving the information-theoretically optimal distribution for normally-distributed weights.

2. **Double quantisation** — quantising the quantisation constants themselves to save additional memory.

3. **Paged optimisers** — managing GPU memory spikes by offloading optimiser states to CPU RAM when needed.

The trainable LoRA adapters remain in BF16 (higher precision). Only they are updated; the quantised base model weights are dequantised to BF16 on the fly for each forward pass.

</details>

---

➡️ **Full quiz with 3 questions:** [Knowledge Checks → Fine-Tuning](knowledge-checks.md#8-fine-tuning)

---

## Further Reading

| Resource | Type | Notes |
|---------|------|-------|
| [LoRA paper](https://arxiv.org/abs/2106.09685) | Paper | Original LoRA derivation |
| [QLoRA paper](https://arxiv.org/abs/2305.14314) | Paper | 4-bit fine-tuning with NF4 |
| [DoRA paper](https://arxiv.org/abs/2402.09353) | Paper | Weight-decomposed LoRA |
| [DPO paper](https://arxiv.org/abs/2305.18290) | Paper | Direct preference optimisation |
| [LIMA paper](https://arxiv.org/abs/2305.11206) | Paper | Quality over quantity in instruction data |
| [HuggingFace PEFT](https://huggingface.co/docs/peft) | Documentation | Practical LoRA/PEFT implementation guide |
| [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl) | Documentation | DPO/PPO training library |

---

*Navigation: [← Training Techniques](training-techniques.md) · [Advanced Home](../README.md) · [Next: Evaluation →](evaluation.md)*
