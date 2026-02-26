# Evaluation

> 📍 [Home](../../README.md) › [Advanced](../README.md) › Evaluation

---

## Learning Objectives

By the end of this document you will be able to:

- Describe the most widely used LLM benchmarks and what they measure
- Identify the limitations and biases in automated evaluation
- Explain the LLM-as-judge approach and its known failure modes
- Understand calibration and when to use it
- Design an evaluation pipeline for a practical LLM application

---

## Prerequisites

| Topic | Where to Review |
|-------|----------------|
| What language models are | [Model Architectures](model-architectures.md) |
| Fine-tuning basics | [Fine-Tuning](fine-tuning.md) |

---

## 1. Why Evaluation Is Hard

Evaluating LLMs is fundamentally different from evaluating classical ML models:

- **Open-ended outputs:** Unlike classification, there is often no single correct answer
- **Multi-dimensional quality:** Accuracy, helpfulness, safety, fluency, faithfulness all matter
- **Contamination risk:** Training data may include benchmark answers, inflating scores
- **Distributional shift:** Benchmarks become stale as models improve and test sets leak

---

## 2. Standard Benchmarks

### Knowledge and Reasoning

| Benchmark | Task | Notes |
|-----------|------|-------|
| **MMLU** | 57-domain multiple choice | Most widely used; tests world knowledge |
| **GPQA** | Graduate-level expert Q&A | Hard; designed to be difficult for non-experts |
| **AGIEval** | Human exams (SAT, LSAT, GRE) | Tests reasoning on official standardised tests |
| **ARC** | Grade school science questions | Easy (ARC-Easy) and hard (ARC-Challenge) variants |
| **HellaSwag** | Commonsense completion | Tests world knowledge via sentence completion |

### Coding

| Benchmark | Task | Notes |
|-----------|------|-------|
| **HumanEval** | Python function completion | 164 problems; unit-test verified |
| **MBPP** | Mostly basic Python problems | 374 problems; broader coverage than HumanEval |
| **LiveCodeBench** | Fresh competitive programming problems | Updated regularly to avoid contamination |
| **SWE-bench** | Real GitHub issue resolution | End-to-end coding agent evaluation |

### Mathematical Reasoning

| Benchmark | Task | Notes |
|-----------|------|-------|
| **GSM8K** | Grade school math word problems | Good for lower-capability models |
| **MATH** | Competition mathematics | Highly challenging; 5-level difficulty |
| **AIME** | American Invitational Mathematics Exam | Frontier model evaluation |

### Holistic Evaluation

| Framework | Description |
|-----------|-------------|
| **HELM** | Standardised evaluation across 40+ tasks with transparency requirements |
| **Open LLM Leaderboard (HuggingFace)** | Community standard for comparing open-weight models |
| **BIG-Bench** | 200+ diverse tasks contributed by the research community |

---

## 3. Evaluation Methodology

### Exact Match vs. Generation

**Exact Match (EM):** Compare model output character-for-character against reference.
- Best for factual QA, code (with unit tests), structured outputs
- Brittle: penalises valid paraphrases

**Generation + Scoring:** Generate free-form output; score with a metric or judge.
- Necessary for open-ended tasks
- Introduces scorer variance

### Few-Shot Prompting for Benchmarks

Most benchmarks are evaluated in few-shot or zero-shot settings:

```
System: You are a helpful assistant.

User: Answer the following multiple choice question.
      Q: What is the capital of France?
      A) London B) Paris C) Madrid D) Rome
      Answer: B

User: Answer the following multiple choice question.
      Q: The mitochondria is known as the...
      A) Nucleus B) Cell wall C) Powerhouse D) Membrane
      Answer:
```

### Chain-of-Thought Prompting

Adding "Let's think step by step" significantly improves reasoning benchmark performance. Most modern benchmarks require CoT for frontier model evaluation.

---

## 4. LLM-as-Judge

Using a capable LLM (e.g., GPT-4) to evaluate generated outputs — scaling human judgement automatically.

### Common Evaluation Patterns

**Single-answer grading:** Score a single response (1–10 or pass/fail):

```python
JUDGE_PROMPT = """
You are an impartial evaluator. Score the following response on a scale of 1-10.

Question: {question}
Response: {response}

Scoring criteria:
- Accuracy (0-4 points): Is the information correct?
- Completeness (0-3 points): Does it fully answer the question?
- Clarity (0-3 points): Is the response clear and well-structured?

Output format: {"score": <int>, "reasoning": "<string>"}
"""
```

**Pairwise comparison (tournament-style):** Given two responses, which is better? More reliable than absolute scoring.

### Known Biases in LLM Judges

| Bias | Description | Mitigation |
|------|-------------|------------|
| **Position bias** | Prefers responses in first/second position | Swap order, average |
| **Verbosity bias** | Prefers longer responses regardless of quality | Length-normalised rubrics |
| **Self-preference** | Models prefer their own outputs | Use a different judge model |
| **Sycophancy** | Influenced by claimed authority or flattery | Structured rubrics; blind evaluation |

---

## 5. Calibration

A well-calibrated model is confident when it's correct and uncertain when it might be wrong.

### Expected Calibration Error (ECE)

```
ECE = Σ_b (|B_b| / n) · |accuracy(B_b) - confidence(B_b)|
```

Where bins `B_b` group predictions by confidence level.

A perfectly calibrated model has ECE = 0: when it says "70% confident," it's correct 70% of the time.

### Reliability Diagram

Plot predicted confidence vs actual accuracy across bins. A perfectly calibrated model falls on the diagonal.

**Modern LLMs tend to be overconfident.** Temperature scaling post-hoc calibration is a simple and effective fix.

---

## 6. Data Contamination Detection

Test set contamination (training data includes benchmark questions) artificially inflates scores.

### Methods

**N-gram overlap:** Check if n-grams from the test set appear in the training corpus.

**Canonical form matching:** Normalise whitespace/case and check for exact matches.

**Membership inference:** Use model log-probabilities to detect memorisation.

```python
# Contamination probe: compare log-probs on original vs corrupted benchmark
original_logprob = model.score(original_question)
corrupted_logprob = model.score(shuffled_choices_question)

# If original >> corrupted, model may have memorised this example
contamination_score = original_logprob - corrupted_logprob
```

### Mitigations

- Use **private test sets** not published on the web
- Use **dynamic benchmarks** with fresh problems (LiveCodeBench)
- Report benchmark dates relative to training cutoff

---

## 7. Evaluating RAG and Agent Systems

### RAG Evaluation

| Metric | Measures | Tool |
|--------|---------|------|
| **Context precision** | Are retrieved chunks relevant? | RAGAS |
| **Context recall** | Are all relevant chunks retrieved? | RAGAS |
| **Answer faithfulness** | Does the answer stay grounded in context? | RAGAS |
| **Answer relevancy** | Does the answer address the question? | RAGAS |

### Agent Evaluation

Agents require end-to-end evaluation on task completion:

| Benchmark | Domain | Notes |
|-----------|--------|-------|
| **SWE-bench** | Software engineering | Resolves real GitHub issues |
| **OSWorld** | Desktop computer use | GUI interaction tasks |
| **WebArena** | Web browsing tasks | Structured web tasks |
| **τ-bench** | Tool use | Agentic function calling |

Key dimensions: task success rate, efficiency (steps to completion), tool call accuracy.

---

## 8. Human Evaluation

Despite the cost, human evaluation remains the gold standard for quality, safety, and preference.

### Crowdworker Guidelines

- Provide detailed annotation rubrics with examples of each score level
- Include quality checks (gold-standard items with known answers)
- Pilot with small batches to calibrate annotators
- Compute inter-annotator agreement (Fleiss' κ or Krippendorff's α)

### Side-by-Side (SxS) Evaluation

Present two responses simultaneously and ask annotators to:
1. Select the better response, OR
2. Rate both independently and compare

SxS is more reliable than absolute ratings for detecting small differences.

---

## Key Concepts Summary

| Concept | One-line description |
|---------|---------------------|
| **MMLU** | Multi-domain multiple choice knowledge benchmark |
| **LLM-as-judge** | Use a strong model to score or rank generated outputs |
| **Calibration** | How well model confidence matches actual accuracy |
| **ECE** | Expected Calibration Error — quantifies miscalibration |
| **Contamination** | Training data includes test examples; inflates scores |
| **RAGAS** | Framework for evaluating RAG pipeline quality |

---

## Further Reading

| Resource | Type | Notes |
|---------|------|-------|
| [HELM paper](https://arxiv.org/abs/2211.09110) | Paper | Holistic evaluation framework |
| [Chatbot Arena (LMSYS)](https://lmsys.org/blog/2023-05-03-arena/) | Blog | Human preference leaderboard via pairwise battles |
| [LLM-as-a-Judge paper](https://arxiv.org/abs/2306.05685) | Paper | MT-Bench and LLM judge methodology |
| [RAGAS](https://docs.ragas.io/) | Documentation | RAG evaluation framework |
| [Contamination survey](https://arxiv.org/abs/2310.18018) | Paper | Systematic review of benchmark contamination |
| [Calibration paper](https://arxiv.org/abs/1706.04599) | Paper | Guo et al. on neural network calibration |

---

*Navigation: [← Fine-Tuning](fine-tuning.md) · [Advanced Home](../README.md) · [Next: RAG →](rag.md)*
