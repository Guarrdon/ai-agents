# Advanced AI Section

> 📍 [Home](../README.md) › Advanced AI Section

Welcome to the **Advanced AI** section of the AI Learning Resources. This section is designed for developers, researchers, and technical professionals who are ready to go beyond the basics and explore the mechanics, architectures, and implementation details behind modern AI systems.

**[← Back to Home](../README.md)**

> 💡 **New to AI?** If you haven't already, we recommend starting with the [Beginner Section](../beginner/README.md) to build foundational understanding before diving into the technical content here.

---

## Who Is This For?

Before diving in, make sure you're comfortable with the following:

- [ ] Basic Python programming (functions, classes, loops)
- [ ] Foundational linear algebra (vectors, matrix multiplication, dot products)
- [ ] Basic calculus (derivatives, chain rule)
- [ ] Conceptual understanding of what machine learning is (see the [Beginner section](../beginner/docs/what-is-ai.md) if needed)
- [ ] Familiarity with at least one ML framework (PyTorch, TensorFlow, or JAX)

If you can check most of the boxes above, you're in the right place!

---

## 🗺️ Learning Path

The topics below are ordered to build on each other. We recommend following them in sequence, though each document is also designed to stand alone as a reference.

```
1. Neural Networks          → The foundation of modern deep learning
2. Model Architectures      → Transformers, attention, SSMs, MoE
3. Training Techniques      → Optimisation, distributed training, precision
4. Fine-Tuning              → PEFT, LoRA, instruction tuning, RLHF
5. Evaluation               → Benchmarks, metrics, LLM-as-judge
6. Retrieval-Augmented Gen  → RAG pipelines, indexing, re-ranking
7. AI Agents                → Tool use, planning, memory, multi-agent
8. Safety & Alignment       → RLHF, interpretability, alignment research
```

---

## 📚 Topics

### 1. Neural Networks
**File:** [docs/neural-networks.md](docs/neural-networks.md)

The mathematical and computational foundations of deep learning — perceptrons, activation functions, backpropagation, optimisation, and regularisation. Understanding these fundamentals is essential for everything that follows.

> **Covers:** Feedforward networks, activation functions, backpropagation, gradient descent variants, regularisation techniques, batch normalisation

---

### 2. Model Architectures
**File:** [docs/model-architectures.md](docs/model-architectures.md)

A deep dive into the Transformer architecture and its modern variants — multi-head attention, position encodings, mixture-of-experts, state space models, and the scaling laws that govern their behaviour.

> **Covers:** Transformer internals, attention variants (MHA, MQA, GQA, FlashAttention), position encodings (RoPE, ALiBi), MoE, SSMs (Mamba), scaling laws

---

### 3. Training Techniques
**File:** [docs/training-techniques.md](docs/training-techniques.md)

Practical and theoretical techniques for training large models — advanced optimisers, learning rate schedules, distributed training strategies, mixed-precision, and training stability.

> **Covers:** AdamW, Lion, WSD schedules, data pipelines, FSDP, tensor/pipeline parallelism, FP8 training, loss spike debugging, μP parametrisation

---

### 4. Fine-Tuning
**File:** [docs/fine-tuning.md](docs/fine-tuning.md)

Adapting pre-trained models to specific tasks — full fine-tuning, parameter-efficient methods (LoRA, DoRA, GaLore), instruction tuning, and alignment fine-tuning with DPO and its variants.

> **Covers:** Full fine-tuning, LoRA/QLoRA/DoRA, PEFT comparison, RLHF, DPO/ORPO/SimPO, reward modelling, weight merging

---

### 5. Evaluation
**File:** [docs/evaluation.md](docs/evaluation.md)

Rigorous evaluation of language models — standard benchmarks, custom metrics, LLM-as-judge frameworks, human evaluation, and detecting data contamination.

> **Covers:** MMLU, HELM, GPQA, LLM-as-judge bias, calibration/ECE, contamination detection, self-consistency, agent evaluation

---

### 6. Retrieval-Augmented Generation (RAG)
**File:** [docs/rag.md](docs/rag.md)

Building production RAG systems — dense and sparse retrieval, re-ranking, advanced architectures (RAPTOR, GraphRAG, ColPali), and corrective/self-RAG patterns.

> **Covers:** Chunking, embedding models, FAISS/HNSW, ColBERT, SPLADE, re-ranking, RAPTOR, GraphRAG, agentic RAG, prompt injection defence

---

### 7. AI Agents
**File:** [docs/agents.md](docs/agents.md)

LLM-based autonomous agents — tool use, planning strategies, memory systems, multi-agent coordination, and production considerations.

> **Covers:** ReAct, MCTS, Reflexion, tool use, episodic/semantic memory, multi-agent patterns, Computer Use, MCP, SWE-bench

---

### 8. Safety & Alignment
**File:** [docs/safety-alignment.md](docs/safety-alignment.md)

The technical foundations of AI safety and alignment — RLHF, Constitutional AI, mechanistic interpretability, red-teaming, and scalable oversight research.

> **Covers:** Alignment taxonomy, RLAIF, DPO derivation, SAE interpretability, activation steering, GCG attacks, automated red-teaming, scalable oversight

---

## 🧩 How Each Document Is Structured

Every topic document in this section follows a consistent layout:

| Section | Purpose |
|---------|---------|
| **Breadcrumb navigation** | Shows your location in the hierarchy |
| **Learning objectives** | What you will understand after reading |
| **Prerequisites** | Specific prior knowledge assumed |
| **Core content** | Technical deep-dive with code and math |
| **Key concepts summary** | Quick-reference table of terms |
| **Further reading** | Papers, docs, and tutorials to go deeper |
| **Navigation footer** | Links to previous and next topics |

---

## 💡 Tips for Reading This Section

- **Code examples** are in Python, using PyTorch unless otherwise noted.
- **Mathematical notation** uses LaTeX-style fencing — render with a Markdown viewer that supports it (e.g., GitHub, VS Code with extensions, Obsidian).
- **Each document can be read independently** as a reference — you don't have to follow the learning path in order.
- **Further reading** sections link to original papers — don't skip them; reading primary sources is a skill worth building.

---

*Last updated: February 2026*
