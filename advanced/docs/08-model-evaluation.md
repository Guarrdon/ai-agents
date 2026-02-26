> 📍 [Home](../../README.md) › [Advanced](../README.md) › Model Evaluation Metrics

# Model Evaluation Metrics

> **Learning Objectives**
>
> By the end of this document, you will be able to:
> - Compute and interpret classification metrics: accuracy, precision, recall, F1, AUC-ROC, and MCC
> - Select the right evaluation metric for imbalanced, multi-class, and multi-label settings
> - Apply regression evaluation metrics (MAE, MSE, RMSE, R², Huber loss)
> - Evaluate language model quality using perplexity, BLEU, ROUGE, BERTScore, and human evaluation frameworks
> - Implement statistical significance testing for fair model comparisons
> - Design robust evaluation protocols: data splits, leakage prevention, and benchmark construction
> - Apply calibration diagnostics to measure whether a model's confidence matches its accuracy

---

## 1. Classification Metrics

### 1.1 The Confusion Matrix

All standard classification metrics derive from the confusion matrix. For a binary classifier:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

```python
from sklearn.metrics import confusion_matrix
import numpy as np

y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 0])

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
# TP=3, FP=1, TN=4, FN=2
```

### 1.2 Core Classification Metrics

#### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Limitation:** Misleading on imbalanced datasets. A classifier predicting "negative" for every example achieves 95% accuracy when the positive class is only 5% of the data.

#### Precision

```
Precision = TP / (TP + FP)
```

"Of all the examples predicted positive, what fraction actually were positive?"

High precision is prioritized when false positives are costly (e.g., spam detection — you don't want to delete legitimate emails).

#### Recall (Sensitivity, True Positive Rate)

```
Recall = TP / (TP + FN)
```

"Of all the actual positives, what fraction did we correctly identify?"

High recall is prioritized when false negatives are costly (e.g., cancer screening — missing a diagnosis is worse than a false alarm).

#### F1 Score

The harmonic mean of precision and recall — a single metric that balances both:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

The harmonic mean is used rather than the arithmetic mean because it penalizes extreme imbalance between precision and recall.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
```

#### F-beta Score

Generalizes F1 by weighting recall β times more important than precision:

```
F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

- `β = 0.5`: Precision-weighted (FP more costly than FN)
- `β = 1`: Balanced F1
- `β = 2`: Recall-weighted (FN more costly than FP)

### 1.3 Threshold-Independent Metrics

#### ROC Curve and AUC-ROC

The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (recall) against the False Positive Rate at every possible classification threshold.

```
False Positive Rate (FPR) = FP / (FP + TN)
```

The Area Under the ROC Curve (AUC-ROC) summarizes the curve in a single number:
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier (diagonal line)
- AUC < 0.5: Worse than random (predicting the wrong class)

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

y_scores = np.array([0.9, 0.8, 0.3, 0.1, 0.2, 0.7, 0.95, 0.15, 0.4, 0.25])

auc = roc_auc_score(y_true, y_scores)
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
```

**AUC-ROC interpretation:** AUC equals the probability that a randomly chosen positive example is scored higher than a randomly chosen negative example.

#### Precision-Recall Curve and AUC-PR

More informative than ROC on heavily imbalanced datasets. Plots precision (y-axis) against recall (x-axis) across thresholds.

```python
from sklearn.metrics import average_precision_score, precision_recall_curve

ap = average_precision_score(y_true, y_scores)  # Area under P-R curve
precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
```

**Rule of thumb:** Use AUC-ROC when class imbalance is moderate (< 10:1). Prefer AUC-PR when imbalance is severe.

### 1.4 Matthews Correlation Coefficient (MCC)

MCC provides a more balanced single-value metric for binary classification, even on imbalanced datasets:

```
MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

Ranges from -1 (perfectly wrong) to +1 (perfectly correct); 0 is random.

```python
from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_true, y_pred)
```

### 1.5 Multi-Class Classification

For `C`-class problems, averaging strategies determine how per-class metrics are aggregated:

| Strategy | Description | Use When |
|---|---|---|
| **Macro avg** | Average metric over all classes equally | Class distribution matters equally |
| **Micro avg** | Pool all TP/FP/FN before computing | Each example matters equally (similar to accuracy) |
| **Weighted avg** | Weight classes by their support (count) | Imbalanced classes; want per-class metric weighted by frequency |

```python
from sklearn.metrics import classification_report

y_true_mc = [0, 1, 2, 0, 1, 2, 0, 1]
y_pred_mc = [0, 1, 1, 0, 2, 2, 0, 1]

print(classification_report(y_true_mc, y_pred_mc, target_names=["Cat", "Dog", "Bird"]))
```

---

## 2. Regression Metrics

### 2.1 Core Regression Metrics

#### Mean Absolute Error (MAE)

```
MAE = (1/n) Σ |y_i - ŷ_i|
```

Interpretable in the same units as the target variable. Robust to outliers (linear penalty).

#### Mean Squared Error (MSE) and Root MSE (RMSE)

```
MSE  = (1/n) Σ (y_i - ŷ_i)²
RMSE = √MSE
```

MSE penalizes large errors quadratically, making it sensitive to outliers. RMSE has the same units as the target.

#### Mean Absolute Percentage Error (MAPE)

```
MAPE = (100/n) Σ |y_i - ŷ_i| / |y_i|
```

Scale-independent. Undefined when `y_i = 0`; unstable when `y_i` is near zero.

#### R² (Coefficient of Determination)

```
R² = 1 - SS_res / SS_tot   where   SS_res = Σ(y_i - ŷ_i)²,  SS_tot = Σ(y_i - ȳ)²
```

Proportion of variance in the target explained by the model. `R² = 1` is perfect; `R² = 0` means the model predicts the mean for all examples; `R² < 0` means the model is worse than predicting the mean.

**Adjusted R²** penalizes for the number of features, preventing artificial inflation from adding irrelevant predictors.

#### Huber Loss

Combines MAE and MSE: MSE-like for small errors (under threshold `δ`), MAE-like for large errors (more robust to outliers):

```
L_δ(y, ŷ) = 0.5(y - ŷ)²          if |y - ŷ| ≤ δ
           = δ|y - ŷ| - 0.5δ²     otherwise
```

```python
import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    residuals = np.abs(y_true - y_pred)
    return np.where(
        residuals <= delta,
        0.5 * residuals**2,
        delta * residuals - 0.5 * delta**2
    ).mean()
```

---

## 3. Language Model Evaluation Metrics

### 3.1 Perplexity

Perplexity measures how well a language model predicts a held-out text corpus. Lower perplexity indicates better predictions.

```
Perplexity = exp(-1/N Σ log P(w_i | w_1, ..., w_{i-1}))
           = exp(cross-entropy)
```

A perplexity of `k` means the model is as uncertain as if it had to choose uniformly among `k` words at each step.

```python
import torch
import torch.nn.functional as F

def compute_perplexity(model, tokenizer, text: str, device="cuda") -> float:
    encodings = tokenizer(text, return_tensors="pt").to(device)
    input_ids = encodings.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    # outputs.loss is the mean cross-entropy over tokens
    return torch.exp(outputs.loss).item()
```

**Limitation:** Perplexity only measures fit to the reference text distribution, not output quality on downstream tasks. Two models with the same perplexity can differ greatly in usefulness.

### 3.2 BLEU (Bilingual Evaluation Understudy)

BLEU measures n-gram overlap between model output and reference translations. Originally designed for machine translation; widely adopted for other generation tasks.

```
BLEU = BP × exp(Σ_n w_n log p_n)

where:
  p_n = precision of n-gram matches (clipped against reference counts)
  w_n = weight for n-gram order n (typically uniform: 1/N)
  BP  = brevity penalty = min(1, exp(1 - reference_length/hypothesis_length))
```

```python
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

references = [["the cat sat on the mat".split()]]
hypotheses = ["the cat is on the mat".split()]

# BLEU-4 with smoothing (prevents 0 for short outputs)
smoother = SmoothingFunction().method1
bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smoother)
print(f"BLEU-4: {bleu4:.4f}")
```

**Limitations:**
- Only measures surface form overlap; misses paraphrases
- Correlates poorly with human judgments on summarization and dialogue
- Brevity penalty is a crude substitute for recall
- Requires a reference; poor for open-ended generation

### 3.3 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Primarily used for summarization evaluation. Unlike BLEU (precision-focused), ROUGE is recall-focused.

| Variant | Description |
|---|---|
| **ROUGE-N** | N-gram recall between hypothesis and reference |
| **ROUGE-L** | Longest Common Subsequence (LCS) F-measure |
| **ROUGE-LSum** | ROUGE-L at summary level (by newline boundaries) |

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
reference = "The cat sat on the mat near the window."
hypothesis = "A cat was sitting on the mat."

scores = scorer.score(reference, hypothesis)
for metric, score in scores.items():
    print(f"{metric}: P={score.precision:.3f}, R={score.recall:.3f}, F={score.fmeasure:.3f}")
```

### 3.4 BERTScore

BERTScore (Zhang et al., 2019) computes precision, recall, and F1 over contextual token embeddings (from a BERT-like model) rather than exact n-gram matches.

```
P_BERT = (1/|x̂|) Σ_{x̂_i ∈ x̂} max_{x_j ∈ x} cos(e_i, e_j)
R_BERT = (1/|x|)  Σ_{x_j ∈ x}  max_{x̂_i ∈ x̂} cos(e_i, e_j)
F_BERT = 2 × P_BERT × R_BERT / (P_BERT + R_BERT)
```

```python
from bert_score import score as bertscore

hypotheses = ["The cat is on the mat."]
references = ["A cat sat on the mat."]

P, R, F1 = bertscore(hypotheses, references, lang="en", model_type="roberta-large")
print(f"BERTScore F1: {F1.mean():.4f}")
```

**Advantage:** Captures semantic similarity beyond exact word overlap. More correlated with human judgments than BLEU/ROUGE.

### 3.5 Evaluation Benchmarks for LLMs

| Benchmark | Focus | Format |
|---|---|---|
| **MMLU** | Multitask language understanding (57 subjects) | 4-choice MCQ |
| **HellaSwag** | Commonsense NLI; sentence completion | 4-choice |
| **TruthfulQA** | Factual accuracy; avoiding plausible falsehoods | Generation |
| **GSM8K** | Grade school math; multi-step reasoning | Generation |
| **HumanEval** | Python code generation | Functional correctness |
| **BIG-Bench Hard** | 23 challenging reasoning tasks | Various |
| **MT-Bench** | Multi-turn dialogue; GPT-4 judge scores responses | Generation + LLM judge |
| **HELM** | Holistic evaluation across 42 scenarios | Multi-metric |

**Benchmark contamination:** A critical concern — if benchmark data appears in a model's training corpus, scores are inflated and comparisons are invalid. Evaluators must check for training set overlap when reporting results.

### 3.6 Human Evaluation Frameworks

Automated metrics correlate imperfectly with human judgments, especially for open-ended generation. Human evaluation remains the gold standard but is expensive and slow.

**Standard rating dimensions:**

| Dimension | Description | Scale |
|---|---|---|
| **Fluency** | Grammaticality and natural language flow | 1–5 |
| **Coherence** | Logical consistency and structure | 1–5 |
| **Faithfulness** | Factual accuracy w.r.t. a provided context (critical for summarization and RAG) | 1–5 |
| **Relevance** | Addresses the prompt or question | 1–5 |
| **Helpfulness** | Practical value of the response | 1–5 |
| **Harmlessness** | Absence of toxic, biased, or dangerous content | Binary + category |

**Pairwise comparisons:** Rather than absolute rating scales (subject to annotator calibration differences), pairwise comparisons ("Which response is better: A or B?") are more reliable and reproducible.

**Inter-annotator agreement:** Measure with Cohen's κ (two annotators) or Krippendorff's α (multiple annotators). An α > 0.6 is generally considered acceptable for subjective tasks.

### 3.7 LLM-as-Judge

Using a powerful LLM (e.g., GPT-4) as an automated evaluator. The judge model rates outputs on defined criteria and can scale to thousands of examples at a fraction of human evaluation cost.

```python
def llm_judge_prompt(criterion, question, response_a, response_b):
    return f"""You are an impartial judge evaluating AI assistant responses.

Criterion: {criterion}

Question: {question}

Response A: {response_a}

Response B: {response_b}

Which response better satisfies the criterion? Reply with "A", "B", or "Tie" and a brief justification."""
```

**Known biases in LLM judges:**
- **Position bias:** Prefers the first response presented
- **Verbosity bias:** Prefers longer responses regardless of quality
- **Self-preference:** A model tends to prefer its own outputs

**Mitigation:** Swap the order of responses and average; use an ensemble of judges; calibrate against human ratings.

---

## 4. Calibration

A well-calibrated model's predicted probabilities match empirical frequencies: when a classifier predicts 70% confidence, approximately 70% of those predictions should be correct.

### 4.1 Reliability Diagrams and Expected Calibration Error (ECE)

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return ece

def plot_reliability_diagram(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    accuracies = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            accuracies.append(np.nan)
        else:
            accuracies.append(y_true[mask].mean())

    plt.figure(figsize=(5, 5))
    plt.bar(bin_centers, accuracies, width=0.1, align="center", alpha=0.7, label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
```

### 4.2 Calibration Methods

| Method | Description | When to Use |
|---|---|---|
| **Temperature scaling** | Divide logits by a learned temperature `T`; scales confidence without changing predictions | Post-hoc calibration for neural networks; fast and effective |
| **Platt scaling** | Fit a logistic regression on held-out calibration data | Binary classifiers with limited calibration data |
| **Isotonic regression** | Non-parametric monotone calibration | Larger calibration sets; more flexible than Platt |
| **Histogram binning** | Assign calibrated probabilities per confidence bin | Simple; works well in practice |

```python
class TemperatureScaling:
    """Post-hoc calibration via temperature scaling."""
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, labels):
        import scipy.optimize as opt
        def nll(T):
            scaled = logits / T[0]
            log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
            return -log_probs[np.arange(len(labels)), labels].mean()
        result = opt.minimize(nll, [1.5], bounds=[(0.1, 10.0)])
        self.temperature = result.x[0]

    def calibrate(self, logits):
        return logits / self.temperature
```

---

## 5. Evaluation Protocol and Experimental Design

### 5.1 Data Splits

Proper separation of data is critical for unbiased evaluation.

| Split | Purpose | Typical Size |
|---|---|---|
| **Training** | Model parameter optimization | 60–80% |
| **Validation** | Hyperparameter tuning; early stopping | 10–20% |
| **Test** | Final unbiased evaluation; report once | 10–20% |

**Never tune hyperparameters on the test set.** Each use of the test set for decision-making introduces bias; only report the test metric once at the end of all development.

### 5.2 Cross-Validation

K-fold cross-validation uses the entire dataset for evaluation while maintaining test-set hygiene:

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

def cross_validate(model_class, X, y, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = model_class()
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, average="macro"))
    return np.array(scores)

scores = cross_validate(MyModel, X, y, k=5)
print(f"CV F1: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Stratified K-fold** preserves the class distribution within each fold — essential for imbalanced datasets.

### 5.3 Avoiding Data Leakage

Data leakage occurs when information from the test set influences model training or selection, inflating performance estimates.

**Common leakage sources:**
- **Feature leakage:** A feature encodes the label (e.g., including the timestamp of a fraud event as a feature)
- **Preprocessing leakage:** Fitting a scaler or imputer on the full dataset before splitting
- **Temporal leakage:** Using future data to predict past events in time-series

```python
# WRONG: Fit scaler on entire dataset (leaks test statistics into training)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Don't do this before splitting!

# CORRECT: Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)   # Fit + transform on train
X_test_scaled  = scaler.transform(X_test)         # Transform only on test
```

### 5.4 Statistical Significance Testing

When comparing two models, use statistical tests to determine whether observed differences are genuine or due to sampling variation.

**McNemar's test** for paired binary classification on the same test set:

```python
from statsmodels.stats.contingency_tables import mcnemar

# b = model_A correct, model_B wrong
# c = model_A wrong, model_B correct
contingency_table = np.array([[a, b], [c, d]])
result = mcnemar(contingency_table, exact=False, correction=True)
print(f"p-value: {result.pvalue:.4f}")
# p < 0.05 → statistically significant difference
```

**Bootstrap confidence intervals:**

```python
def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=0.95):
    n = len(y_true)
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        score = metric_fn(y_true[idx], y_pred[idx])
        bootstrap_scores.append(score)
    lower = np.percentile(bootstrap_scores, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_scores, (1 + ci) / 2 * 100)
    return np.mean(bootstrap_scores), lower, upper
```

### 5.5 Error Analysis

Aggregate metrics hide where a model fails. Systematic error analysis drives effective improvements.

**Sliced evaluation:** Compute metrics separately for meaningful subgroups (by age, gender, geography, input length, topic category) to detect differential performance.

```python
def sliced_evaluation(y_true, y_pred, groups, metric_fn):
    unique_groups = np.unique(groups)
    results = {}
    for group in unique_groups:
        mask = groups == group
        results[group] = {
            "n": mask.sum(),
            "metric": metric_fn(y_true[mask], y_pred[mask])
        }
    return results
```

**Confusion analysis:** For multi-class problems, examine which classes are most confused with one another to guide data collection or feature engineering.

---

## 6. Specialized Metrics

### 6.1 Object Detection

| Metric | Description |
|---|---|
| **IoU (Intersection over Union)** | Overlap between predicted and ground-truth bounding boxes: `IoU = Area(A∩B) / Area(A∪B)` |
| **mAP (mean Average Precision)** | Average precision over all IoU thresholds and classes; COCO standard: mAP@[0.5:0.05:0.95] |
| **mAP@50** | mAP at IoU threshold 0.50; PASCAL VOC standard |

### 6.2 Ranking and Recommendation

| Metric | Description |
|---|---|
| **NDCG@k** | Normalized Discounted Cumulative Gain; accounts for position and graded relevance |
| **MRR** | Mean Reciprocal Rank; reciprocal of the rank of the first relevant result |
| **Precision@k** | Fraction of top-k recommendations that are relevant |
| **Recall@k** | Fraction of all relevant items appearing in top-k |

### 6.3 Fairness Metrics

(Detailed coverage in `06-ai-ethics.md`)

| Metric | Description |
|---|---|
| **Demographic parity difference** | `|P(ŷ=1|A=0) - P(ŷ=1|A=1)|` |
| **Equalized odds difference** | Max difference in TPR and FPR across groups |
| **Predictive parity** | Equal precision across groups |
| **Disparate impact ratio** | `min(P(ŷ=1|A=0), P(ŷ=1|A=1)) / max(...)` |

---

## Further Reading

- **BERTScore** — Zhang, T. et al. (2019). *BERTScore: Evaluating Text Generation with BERT.* [arXiv:1904.09675](https://arxiv.org/abs/1904.09675)
- **BLEU** — Papineni, K. et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation.* ACL 2002.
- **ROUGE** — Lin, C. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL Workshop.
- **MT-Bench / Chatbot Arena** — Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
- **MMLU Benchmark** — Hendrycks, D. et al. (2021). *Measuring Massive Multitask Language Understanding.* [arXiv:2009.03300](https://arxiv.org/abs/2009.03300)
- **Temperature Scaling** — Guo, C. et al. (2017). *On Calibration of Modern Neural Networks.* ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
- **HELM** — Liang, P. et al. (2022). *Holistic Evaluation of Language Models.* [arXiv:2211.09110](https://arxiv.org/abs/2211.09110)
- **Scikit-learn Metrics Documentation** — [scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

*Navigation: [← Advanced AI Training Techniques](07-training-techniques.md) · [Advanced Home](../README.md) · [See also: Evaluation](evaluation.md)*

*Last updated: February 2026*
