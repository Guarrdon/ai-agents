# Evaluation

> 📍 [Home](../../README.md) › [Advanced](../README.md) › Evaluation

---

## Learning Objectives

By the end of this document you will be able to:

- Compute and interpret classification metrics: accuracy, precision, recall, F1, AUC-ROC, and MCC
- Apply regression evaluation metrics (MAE, MSE, RMSE, R²)
- Evaluate language model outputs using perplexity, BLEU, ROUGE, and BERTScore
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
| **F1 score** | Harmonic mean of precision and recall |
| **AUC-ROC** | Area under ROC curve; probability a positive outscores a negative |
| **Perplexity** | Exponentiated cross-entropy — how well a model predicts held-out text |
| **BLEU** | N-gram precision metric for text generation (primarily MT) |
| **ROUGE** | Recall-oriented n-gram metric for summarisation |
| **BERTScore** | Semantic similarity using contextual embeddings |

---

## 9. Classical ML Metrics

The metrics above focus on LLM evaluation. This section covers foundational metrics applicable to classification, regression, and generation tasks — essential for evaluating traditional and fine-tuned models.

### 9.1 Classification Metrics

All standard classification metrics derive from the **confusion matrix**:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

```python
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 0])

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred)
```

**Key formulas:**

```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)   — "of all predicted positives, how many were correct?"
Recall    = TP / (TP + FN)   — "of all actual positives, how many did we find?"
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

**AUC-ROC** summarises classifier performance across all thresholds: `AUC = 1.0` is perfect; `AUC = 0.5` is random. Use AUC-PR instead when class imbalance is severe.

**Matthews Correlation Coefficient (MCC):** More balanced than accuracy on imbalanced datasets:
```
MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

### 9.2 Regression Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| **MAE** | `(1/n) Σ |y - ŷ|` | Interpretable; robust to outliers |
| **MSE** | `(1/n) Σ (y - ŷ)²` | Penalises large errors more |
| **RMSE** | `√MSE` | Same units as target |
| **R²** | `1 - SS_res/SS_tot` | Fraction of variance explained; 1 = perfect |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae  = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2   = r2_score(y_true, y_pred)
```

### 9.3 Language Generation Metrics

**Perplexity:** How well a language model predicts held-out text. Lower is better.
```
Perplexity = exp(-1/N Σ log P(w_i | context))
```

**BLEU:** N-gram precision between hypothesis and reference (common for machine translation):
```python
from nltk.translate.bleu_score import corpus_bleu
bleu4 = corpus_bleu(references, hypotheses)
```
Limitation: Only measures surface form; misses paraphrases.

**ROUGE:** Recall-oriented metric for summarisation (ROUGE-1, ROUGE-2, ROUGE-L):
```python
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
scores = scorer.score(reference, hypothesis)
```

**BERTScore:** Semantic similarity via contextual embeddings — more correlated with human judgements than BLEU/ROUGE:
```python
from bert_score import score as bertscore
P, R, F1 = bertscore(hypotheses, references, lang="en", model_type="roberta-large")
```

### 9.4 Evaluation Protocol

| Concern | Best Practice |
|---------|-------------|
| **Data splits** | Train / Validation / Test — never tune on test set |
| **Cross-validation** | Use stratified K-fold for imbalanced datasets |
| **Data leakage** | Fit preprocessors on training data only |
| **Statistical significance** | McNemar's test or bootstrap CIs for model comparisons |
| **Sliced evaluation** | Report metrics separately for meaningful subgroups |

---

## 🧠 Knowledge Check

Test your understanding. Attempt each question before revealing the answer.

**Q1.** A spam classifier achieves 99% accuracy on a dataset where 99% of emails are legitimate. What does this tell you about the model's performance?

<details>
<summary>Answer</summary>

Almost nothing useful. A trivial classifier that always predicts "legitimate" achieves the same 99% accuracy — without detecting a single spam email.

For **class-imbalanced** problems like spam detection, fraud detection, or medical diagnosis, accuracy is misleading. Use instead:

- **Precision** = TP / (TP + FP): how accurate are spam predictions?
- **Recall** = TP / (TP + FN): what fraction of actual spam is caught?
- **F1 score** = harmonic mean of precision and recall
- **AUC-ROC** or **Precision-Recall AUC** for a threshold-independent view

For spam: you likely want high recall (catch most spam) while tolerating some false positives.

</details>

---

**Q2.** What is LLM-as-judge evaluation, and what are two biases you must account for when using it?

<details>
<summary>Answer</summary>

**LLM-as-judge** uses a powerful LLM (e.g., GPT-4, Claude 3 Opus) to score or compare outputs from another model, replacing or supplementing human evaluation.

**Two key biases:**

1. **Position bias:** When comparing two responses (A vs. B), the judge tends to prefer whichever response appears first in the prompt (primacy effect) or last (recency effect). Mitigation: run both orderings (A vs. B and B vs. A) and average the results.

2. **Length bias:** Judges tend to prefer longer, more verbose responses even when conciseness would be more appropriate. Mitigation: include explicit length guidance in the judge prompt; evaluate both a concise and verbose version.

Other valid answers: self-enhancement bias (judge prefers outputs from similar models), style bias (formatting/confidence affects scores), calibration issues.

</details>

---

**Q3.** Why should you never use the test set for hyperparameter tuning, and what split should you use instead?

<details>
<summary>Answer</summary>

The test set is intended to simulate unseen production data — an unbiased estimate of generalisation performance. If you use it to choose hyperparameters, you are implicitly fitting those choices to the test set, causing **data leakage**: the reported test performance will be overly optimistic and not representative of true deployment performance.

Use the **validation set** (also called development set) for all hyperparameter tuning and model selection decisions. The test set should be evaluated only **once**, after all decisions are finalised, to report the final performance.

For small datasets where a held-out validation set would be too small: use **k-fold cross-validation** on the training set to tune hyperparameters, then train on all training data and evaluate once on the test set.

</details>

---

➡️ **Full quiz with 3 questions:** [Knowledge Checks → Evaluation](knowledge-checks.md#9-evaluation)

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
| [BERTScore paper](https://arxiv.org/abs/1904.09675) | Paper | Zhang et al. 2019 |
| [BLEU paper](https://aclanthology.org/P02-1040/) | Paper | Papineni et al. 2002 |
| [Model Evaluation Metrics (In Depth)](08-model-evaluation.md) | Supplementary | Extended coverage of all metrics in this section |

---

*Navigation: [← Fine-Tuning](fine-tuning.md) · [Advanced Home](../README.md) · [Next: AI Ethics →](ethics.md)*
