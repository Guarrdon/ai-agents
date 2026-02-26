# Large Language Models (LLMs)

> 📍 [Home](../../README.md) › [Advanced](../README.md) › Large Language Models

---

## Learning Objectives

By the end of this document you will be able to:

- Explain what makes a language model "large" and describe scaling laws
- Describe the full training pipeline from pretraining through alignment
- Understand tokenisation schemes (BPE, WordPiece, SentencePiece) and their implications
- Explain context windows, position extrapolation, and retrieval-augmented generation
- Apply inference optimisations including quantisation, KV caching, and speculative decoding
- Navigate the landscape of major LLMs (open and closed)

---

## Prerequisites

| Topic | Where to Review |
|-------|----------------|
| Transformer architecture and self-attention | [Model Architectures](model-architectures.md) |
| Backpropagation and gradient descent | [Neural Networks](neural-networks.md) |
| Fine-tuning and RLHF | [Fine-Tuning](fine-tuning.md) |

---

## 1. What Makes a Language Model "Large"?

A **language model** is a probabilistic model over sequences of tokens — it assigns a probability to the next token given a context:

```
P(token_t | token_1, token_2, ..., token_{t-1})
```

Modern LLMs are large along three dimensions simultaneously:

| Dimension | Small Model | Large Model |
|-----------|------------|-------------|
| **Parameters** | Millions (1M–100M) | Billions to trillions (1B–1T+) |
| **Training data** | Gigabytes | Tens of terabytes (trillions of tokens) |
| **Compute** | Single GPU, days | Thousands of GPUs, months |

### Emergent Capabilities

A defining characteristic of LLMs is **emergent behaviour** — capabilities that appear at scale but are largely absent in smaller models:

- Multi-step arithmetic reasoning
- Code generation and debugging
- In-context learning from examples
- Chain-of-thought reasoning
- Instruction following without task-specific training

These capabilities are not explicitly programmed; they emerge from pretraining on massive, diverse data.

### Scaling Laws

Scaling laws describe how validation loss decreases predictably as a **power law** with compute, parameters, and data.

**Kaplan et al. (2020):** Loss scales as:

```
L(N, D) = (N_c/N)^α_N + (D_c/D)^α_D
```

Where `N` is parameter count, `D` is training tokens, and `α_N ≈ 0.076`, `α_D ≈ 0.095`.

**Chinchilla (Hoffmann et al., 2022):** The key revision — for a fixed compute budget `C ∝ 6ND`, optimal allocation requires scaling **both** parameters and data equally:

```
N_opt ∝ C^0.5
D_opt ∝ C^0.5
```

**Practical rule of thumb:** Train on approximately **20 tokens per parameter** for compute-optimal pretraining.

> **Example:** GPT-3 (175B parameters) was significantly undertrained by Chinchilla's criteria. A Chinchilla-optimal model at that compute budget would be ~67B parameters trained on ~1.4T tokens.

---

## 2. Tokenisation

LLMs do not operate on characters or words — they operate on **tokens**, sub-word units that balance vocabulary size with granularity.

### Byte Pair Encoding (BPE)

The dominant tokenisation algorithm (used by GPT-2, GPT-3, GPT-4, LLaMA):

**Algorithm:**
1. Start with all characters as the initial vocabulary
2. Count all adjacent pairs of tokens in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until vocabulary reaches target size (e.g., 32K, 50K, 128K)

```python
# Conceptual illustration of BPE
# "lowest lower newest" →
# Initial: l-o-w-e-s-t  l-o-w-e-r  n-e-w-e-s-t
# After merging "e"+"s" → "es": l-o-w-es-t  l-o-w-e-r  n-e-w-es-t
# After merging "es"+"t" → "est": l-o-w-est  l-o-w-e-r  n-e-w-est
# After merging "l"+"o" → "lo": lo-w-est  lo-w-e-r  n-e-w-est
# ... continues until vocab limit

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

text = "The transformer architecture revolutionised natural language processing."
tokens = tokenizer.encode(text)
decoded = tokenizer.convert_ids_to_tokens(tokens)

print(f"Token count: {len(tokens)}")
print(f"Tokens: {decoded}")
# ['The', 'Ġtransformer', 'Ġarchitecture', 'Ġrevolution', 'ised', ...]
```

### WordPiece

Used by BERT and its derivatives. Similar to BPE but selects merges based on maximising likelihood of the training data rather than raw frequency. Uses `##` prefix to mark continuation tokens:

```
"playing" → ["play", "##ing"]
"tokenisation" → ["token", "##isa", "##tion"]
```

### SentencePiece

A language-agnostic tokeniser (used by T5, Gemma, many multilingual models) that treats the input as a raw byte stream, making it robust to whitespace differences across languages.

### Vocabulary Size Trade-offs

| Vocab Size | Pros | Cons |
|-----------|------|------|
| **Small (~32K)** | Smaller embedding table, faster | More tokens per input, unseen word pieces |
| **Large (~128K+)** | Fewer tokens per input, better for code/multilingual | Larger embedding table |

> **Practical note:** Code models often use larger vocabularies (e.g., DeepSeek-Coder uses 100K) to handle diverse programming language syntax efficiently.

---

## 3. Embeddings

### Token Embeddings

Each token ID is mapped to a dense vector via an embedding matrix `E ∈ R^{|V| × d_model}`:

```python
import torch.nn as nn

vocab_size = 32000
d_model = 4096  # LLaMA 2 7B

embedding = nn.Embedding(vocab_size, d_model)
# Shape: (32000, 4096)

# Convert token IDs to embeddings
token_ids = torch.tensor([[1, 2542, 1234, 7]])  # (batch=1, seq_len=4)
embeddings = embedding(token_ids)  # (1, 4, 4096)
```

### Positional Embeddings

Transformers have no inherent notion of order — positional embeddings inject sequence position. Modern LLMs use **Rotary Position Embeddings (RoPE)**, encoding relative rather than absolute positions (see [Model Architectures](model-architectures.md) §4).

### Contextual vs Static Embeddings

| Type | Example | Key Property |
|------|---------|-------------|
| **Static** | word2vec, GloVe | Same vector regardless of context |
| **Contextual** | BERT, GPT | Vector changes based on surrounding tokens |

Contextual embeddings from LLMs have become dominant representations for downstream tasks — "bank" in "river bank" vs "bank account" gets different vectors.

---

## 4. Pretraining

### Objective: Next-Token Prediction

Decoder-only LLMs (GPT family, LLaMA, Mistral) are trained on **causal language modelling**: predict the next token given all previous tokens.

```
Loss = -Σ log P(token_t | token_{<t})
```

This is also called **autoregressive** or **next-token prediction** training.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "The quick brown fox"
inputs = tokenizer(text, return_tensors="pt")

# Forward pass returns logits over vocabulary at each position
outputs = model(**inputs, labels=inputs["input_ids"])

print(f"Training loss: {outputs.loss.item():.4f}")  # Cross-entropy loss
# Perplexity = exp(loss) — lower is better
```

### Masked Language Modelling (MLM)

Encoder-only models (BERT, RoBERTa) use masked language modelling: randomly mask 15% of tokens and predict them. This enables **bidirectional** attention — every token can attend to every other token.

```
Input:  "The [MASK] sat on the [MASK]"
Target: "cat", "mat"
```

MLM produces excellent contextual representations but cannot generate text natively.

### Pretraining Data

Pretraining data quality matters as much as quantity. Typical sources:

| Source | Notes |
|--------|-------|
| Common Crawl | Filtered web text; noisy but enormous |
| Books (Books3, Gutenberg) | High-quality prose, long-range coherence |
| Wikipedia | Factual, structured text |
| GitHub | Code — essential for coding capabilities |
| arXiv, PubMed | Scientific and technical text |
| StackExchange | Q&A, procedural text |

**Key preprocessing steps:**
- Deduplication (exact and near-duplicate removal)
- Quality filtering (language ID, perplexity-based filtering, heuristics)
- Toxic content removal
- Personally identifiable information (PII) redaction

---

## 5. Training Pipeline: From Pretraining to Deployment

Modern LLMs go through multiple training stages:

```
┌──────────────────────────────────────────────────┐
│  Stage 1: Pretraining                             │
│  Objective: Next-token prediction                  │
│  Data: Trillions of tokens from diverse sources   │
│  Output: Base model (powerful but unaligned)      │
└──────────────────────┬───────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────┐
│  Stage 2: Supervised Fine-Tuning (SFT)            │
│  Objective: Imitate high-quality demonstrations   │
│  Data: 10K–1M instruction-response pairs          │
│  Output: Instruction-following model              │
└──────────────────────┬───────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────┐
│  Stage 3: Alignment (RLHF or DPO)                 │
│  Objective: Align with human preferences          │
│  Data: Pairwise human preference judgements       │
│  Output: Helpful, harmless, aligned assistant     │
└──────────────────────────────────────────────────┘
```

### Reinforcement Learning from Human Feedback (RLHF)

RLHF (Christiano et al., 2017) aligns LLM outputs with human preferences:

1. **Collect preference data:** Show humans pairs of responses; they pick the better one
2. **Train a reward model:** Learn to predict which responses humans prefer
3. **Optimise the policy:** Use PPO to maximise reward while staying close to the SFT model (KL penalty)

```python
# Conceptual RLHF training loop
for batch in preference_dataset:
    # Sample two responses from the SFT model
    response_a = sft_model.generate(batch["prompt"])
    response_b = sft_model.generate(batch["prompt"])

    # Human labels which is preferred
    preferred = human_label(response_a, response_b)

    # Train reward model to score preferred response higher
    reward_model.train_step(batch["prompt"], response_a, response_b, preferred)

# Then: PPO loop maximising reward_model(prompt, response)
# with KL penalty to prevent reward hacking
```

### Direct Preference Optimisation (DPO)

DPO (Rafailov et al., 2023) skips the reward model entirely, directly optimising the policy from preference data:

```
L_DPO(π) = -E_{(x, y_w, y_l)} [ log σ( β log(π(y_w|x)/π_ref(y_w|x)) - β log(π(y_l|x)/π_ref(y_l|x)) ) ]
```

Where `y_w` is the preferred response and `y_l` is the less-preferred response. DPO is simpler and more stable than RLHF and has become the dominant alignment method.

---

## 6. Context Windows and Memory

### What Is a Context Window?

The context window defines how many tokens the model can "see" at once — both the input and the generated output share this budget.

| Model | Context Window |
|-------|---------------|
| GPT-3 | 4,096 tokens |
| GPT-4 Turbo | 128,000 tokens |
| Claude 3 | 200,000 tokens |
| Gemini 1.5 Pro | 1,000,000 tokens |

### Why Context Length Is Hard to Extend

Position embeddings are trained on sequences up to a maximum length. Naive extension degrades performance because the model has never seen those position indices.

**Extension techniques:**

| Method | Approach | Models |
|--------|----------|--------|
| **RoPE scaling** | Linearly scale position indices | LLaMA Long |
| **YaRN** | Non-uniform scaling + attention temperature | Mistral long-context |
| **LongRoPE** | Progressive training with non-uniform positions | Microsoft's LongRoPE |
| **Sliding window attention** | Attend only to a local window + global tokens | Mistral, Longformer |

### Retrieval-Augmented Generation (RAG)

When the relevant information does not fit in the context window, RAG retrieves relevant chunks from an external knowledge base and injects them into the prompt at inference time.

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│   Query     │ →  │  Retriever   │ →  │  Retrieved   │
│ "What is    │    │  (Dense or   │    │  Chunks      │
│  RoPE?"     │    │   Sparse)    │    │  (context)   │
└─────────────┘    └──────────────┘    └──────────┬───┘
                                                   ↓
                                        ┌──────────────────┐
                                        │  LLM Generator   │
                                        │ (Augmented prompt│
                                        │  + retrieved ctx)│
                                        └──────────────────┘
```

See [Retrieval-Augmented Generation](rag.md) for a deep dive.

---

## 7. Inference and Serving

### Autoregressive Decoding

LLMs generate text token by token. Each forward pass produces a probability distribution over the vocabulary; one token is sampled and appended to the context for the next step:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
output = generator("The transformer architecture", max_new_tokens=50)
print(output[0]["generated_text"])
```

**Autoregressive decoding is inherently sequential** — generating token `t` requires token `t-1`.

### Sampling Strategies

| Strategy | Mechanism | Trade-off |
|----------|-----------|-----------|
| **Greedy** | Always pick the highest probability token | Fast but repetitive |
| **Temperature** | Divide logits by T; T<1 = sharper, T>1 = more random | Simple quality/diversity control |
| **Top-k** | Sample from the k highest probability tokens | Avoids long-tail surprises |
| **Top-p (nucleus)** | Sample from smallest set whose cumulative prob ≥ p | Adaptive vocabulary |
| **Beam search** | Maintain multiple candidate sequences | Better quality, more compute |

```python
# Sampling strategies in practice
output = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.8,       # Slight randomness
    top_p=0.95,            # Nucleus sampling
    top_k=50,              # Also limit to top 50 tokens
    max_new_tokens=200,
    repetition_penalty=1.1,  # Discourage repetition
)
```

### Key-Value (KV) Caching

In autoregressive generation, the attention keys and values for all previous tokens are recomputed at every step without caching — an O(n²) waste. KV caching stores these intermediate states:

```
Without KV cache: Step t requires computing K, V for tokens 1...t every time → O(t²) total
With KV cache:    Step t reuses K, V for tokens 1...t-1; only computes new K, V → O(t) total
```

**Memory cost:** KV cache size scales with `batch_size × seq_len × num_layers × 2 × num_heads × head_dim × dtype`.

For a LLaMA 3 8B model at BF16: ~0.5 GB per 4K context tokens per batch item.

### Quantisation

Quantisation reduces weight precision, trading a small quality loss for large memory and speed gains:

| Format | Bits/weight | Memory (7B) | Quality loss |
|--------|------------|-------------|-------------|
| BF16 | 16 | 14 GB | Baseline |
| INT8 | 8 | 7 GB | Minimal |
| INT4 (GPTQ/AWQ) | 4 | 3.5 GB | Slight |
| INT2 | 2 | ~1.75 GB | Significant |

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Load in 4-bit with NF4 quantisation
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Quantise the quantisation constants too
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B-Instruct",
    quantization_config=quant_config,
)
```

### Speculative Decoding

Speculative decoding uses a small "draft" model to propose multiple tokens ahead, then uses the large model to verify them in a single forward pass (since verification is parallelisable). Expected speedup: 2–4×.

```
Draft model proposes: ["The", "capital", "of", "France", "is"]
Large model verifies all 5 tokens in one pass
Accept prefix that matches; reject and resample from the first mismatch
```

---

## 8. The LLM Landscape

### Major Model Families (as of 2026)

| Model | Organisation | Open? | Strengths |
|-------|-------------|-------|-----------|
| **GPT-4o / o3** | OpenAI | No | Frontier general + reasoning |
| **Claude 3.5 / 3.7** | Anthropic | No | Long context, safety, coding |
| **Gemini 2.0** | Google | No | Multimodal, very long context |
| **LLaMA 3.x** | Meta | Yes (weights) | Community ecosystem, strong baseline |
| **Mistral / Mixtral** | Mistral AI | Yes (most) | Efficient, European |
| **DeepSeek V3 / R1** | DeepSeek | Yes | MoE efficiency, strong reasoning |
| **Qwen 2.5** | Alibaba | Yes | Multilingual, code |
| **Gemma 3** | Google | Yes | Compact, high-quality open models |

### Open vs Closed Trade-offs

| Factor | Closed API | Open Weights |
|--------|-----------|-------------|
| **Setup** | Instant, no infrastructure | Self-host required |
| **Customisation** | Prompt only | Full fine-tuning |
| **Privacy** | Data sent to provider | Data stays local |
| **Cost at scale** | Per-token pricing | Fixed compute cost |
| **Capability ceiling** | Highest (frontier) | Improving rapidly |

---

## 9. How LLMs Build on Foundational Concepts

LLMs are not a break from earlier deep learning — they apply foundational concepts at scale:

| Foundational Concept | LLM Application |
|---------------------|-----------------|
| Feedforward networks ([Neural Networks](neural-networks.md)) | Every transformer block contains an FFN sublayer |
| Backpropagation | LLM pretraining: gradients flow back through billions of parameters |
| Softmax over vocabulary | The output head converts logits to token probabilities |
| Embeddings | Tokens → dense vectors; the input representation |
| Attention mechanism | The core of every transformer layer |
| Cross-entropy loss | The training objective for next-token prediction |
| Gradient descent with Adam | Optimizer for pretraining and fine-tuning |
| Regularisation (dropout, weight decay) | Applied during training to prevent overfitting |

---

## Key Concepts Summary

| Concept | One-line description |
|---------|---------------------|
| **Scaling laws** | Loss decreases predictably with compute, parameters, and data |
| **Chinchilla** | Equal scaling of parameters and tokens is compute-optimal |
| **BPE** | Sub-word tokenisation by iteratively merging frequent pairs |
| **Causal LM** | Predict next token given all previous tokens — the pretraining objective |
| **RLHF** | Align model to human preferences via a reward model and PPO |
| **DPO** | Direct alignment from preference pairs without a reward model |
| **KV cache** | Cache computed attention keys/values to speed up autoregressive generation |
| **Quantisation** | Reduce weight precision (e.g., INT4) to save memory with minimal quality loss |
| **Speculative decoding** | Use a small draft model to propose tokens, verified in parallel |
| **RAG** | Inject retrieved context to extend effective knowledge beyond context window |

---

## 🧠 Knowledge Check

Test your understanding. Attempt each question before revealing the answer.

**Q1.** What does the Chinchilla scaling law suggest about the optimal ratio of training tokens to model parameters?

<details>
<summary>Answer</summary>

Chinchilla (Hoffmann et al., 2022) found that for a given compute budget, the optimal ratio is approximately **20 training tokens per model parameter**. Earlier large models like GPT-3 (175B params, ~300B tokens) were undertrained relative to their size. A compute-optimal 175B model should be trained on ~3.5 trillion tokens. This insight led to smaller, better-trained models like Chinchilla 70B outperforming much larger models trained on less data.

</details>

---

**Q2.** What problem does KV caching solve during autoregressive inference, and what is the trade-off?

<details>
<summary>Answer</summary>

**Problem solved:** Without caching, generating each new token requires recomputing Key and Value projections for all previous tokens — O(n) per step, O(n²) total. KV caching stores the computed K and V tensors for all past positions and reuses them, so only the new token's K/V need computing each step.

**Trade-off:** KV cache memory grows linearly with sequence length and batch size. For a 70B parameter model with 80 attention layers, 8 KV heads, and head dimension 128, a context of 4K tokens requires ~40 GB of KV cache for a batch of 1 — becoming a significant memory bottleneck for long contexts or large batches. Techniques like GQA and sliding window attention reduce this cost.

</details>

---

**Q3.** What are the three stages of the RLHF training pipeline, in order?

<details>
<summary>Answer</summary>

1. **Supervised Fine-Tuning (SFT):** Fine-tune the pretrained base model on a curated dataset of high-quality prompt-response demonstrations from human experts.

2. **Reward Model Training:** Train a separate model to predict human preference scores. Human annotators rank multiple model outputs for each prompt; the reward model learns to assign higher scores to preferred responses.

3. **Reinforcement Learning (PPO):** Fine-tune the SFT model using PPO, using the reward model's scores as the reward signal. A KL divergence penalty prevents the policy from drifting too far from the SFT model (preventing "reward hacking").

</details>

---

➡️ **Full quiz with 4 questions:** [Knowledge Checks → Large Language Models](knowledge-checks.md#4-large-language-models)

---

## Further Reading

| Resource | Type | Notes |
|---------|------|-------|
| [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) | Paper | Original GPT-3 paper introducing in-context learning |
| [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) | Paper | Revised scaling laws |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | Paper | Original OpenAI scaling laws (Kaplan et al.) |
| [RLHF: Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) | Paper | Foundational RLHF work |
| [Direct Preference Optimisation (DPO)](https://arxiv.org/abs/2305.18290) | Paper | DPO derivation and results |
| [LLaMA 3 Technical Report](https://arxiv.org/abs/2407.21783) | Paper | Meta's open LLM architecture and training |
| [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/) | Blog | Excellent visual walkthrough of GPT-style models |
| [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) | Code | Minimal GPT implementation in ~300 lines |

---

*Navigation: [← Model Architectures](model-architectures.md) · [Advanced Home](../README.md) · [Next: Generative AI →](generative-ai.md)*
