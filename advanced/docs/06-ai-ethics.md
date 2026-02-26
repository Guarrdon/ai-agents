# AI Ethics, Safety, Bias, and Responsible AI Development

> **Learning Objectives**
>
> By the end of this document, you will be able to:
> - Identify sources of bias in the ML pipeline and apply mitigation strategies
> - Compute and interpret algorithmic fairness metrics and explain their inherent trade-offs
> - Apply model explainability techniques (SHAP, LIME, attention analysis)
> - Describe privacy-preserving ML methods: differential privacy and federated learning
> - Explain the AI alignment problem and the approaches taken by leading research organizations
> - Evaluate responsible AI frameworks and situate them within the regulatory landscape
> - Measure and report the environmental cost of AI training

---

## 1. Why AI Ethics Is an Engineering Discipline

Ethics in AI is often framed as a philosophical concern, but for practitioners it is primarily an engineering discipline with concrete, measurable deliverables: detecting bias before deployment, building explainable models, protecting user privacy, and ensuring systems behave as intended in deployment.

**The stakes are different from traditional software because:**

1. **Scale:** A biased hiring algorithm can affect millions of job applicants before anyone notices the pattern.
2. **Opacity:** Large neural networks are not directly inspectable; errors may not manifest until deployment.
3. **Automation of harm:** Automated systems replicate and amplify errors at machine speed without human judgment in the loop.
4. **Power asymmetry:** The people most affected by AI systems often have the least visibility into them.

**Historical failures provide concrete motivation:**
- **COMPAS recidivism scoring:** ProPublica (2016) found that the system was twice as likely to falsely flag Black defendants as higher risk for future crime than white defendants.
- **Gender bias in facial recognition:** MIT Media Lab (2018) found error rates for darker-skinned women up to 34.7% — vs. 0.8% for lighter-skinned men — in commercial systems from major tech companies.
- **Hiring algorithm gender bias:** Amazon decommissioned a résumé-screening model in 2018 after discovering it penalized résumés that included the word "women's" (e.g., "women's chess club").

---

## 2. Bias and Fairness

### 2.1 Sources of Bias in the ML Pipeline

Bias is not a single problem; it emerges at multiple stages:

| Stage | Bias Type | Example |
|---|---|---|
| **Data collection** | Sampling bias | Collecting medical data only from academic hospitals; underrepresents rural populations |
| **Data collection** | Historical bias | Loan approval training data reflects historical discrimination; model learns to replicate it |
| **Labeling** | Annotator bias | Toxicity labels depend on annotators' cultural backgrounds; systematic mislabeling |
| **Feature engineering** | Proxy features | Zip code encodes race; using it introduces racial bias even without race as an explicit feature |
| **Model training** | Aggregation bias | Training a single global model when the data distribution differs across subgroups |
| **Evaluation** | Evaluation bias | Reporting only aggregate metrics; masking disparate performance across subgroups |
| **Deployment** | Feedback loops | Predictive policing → more arrests in targeted areas → more training data from those areas |

### 2.2 Algorithmic Fairness Metrics

Consider a binary classifier predicting whether a loan applicant will default, where group `A=0` (e.g., Group A) and `A=1` (e.g., Group B) are the protected attribute:

#### Demographic Parity (Statistical Parity)

The probability of a positive prediction is equal across groups:

```
|P(Ŷ=1 | A=0) - P(Ŷ=1 | A=1)| ≤ ε
```

**Interpretation:** Both groups receive positive predictions at the same rate.

#### Equalized Odds

Both True Positive Rate (TPR) and False Positive Rate (FPR) are equal across groups:

```
|TPR(A=0) - TPR(A=1)| ≤ ε
|FPR(A=0) - FPR(A=1)| ≤ ε
```

**Equal opportunity** is a relaxation: only TPR is equalized.

#### Predictive Parity (Calibration Across Groups)

The precision is equal across groups:

```
P(Y=1 | Ŷ=1, A=0) = P(Y=1 | Ŷ=1, A=1)
```

#### Individual Fairness

Similar individuals should receive similar predictions. Requires a domain-specific similarity metric `d`:

```
|f(x_i) - f(x_j)| ≤ L · d(x_i, x_j)   for all pairs (x_i, x_j)
```

(Lipschitz condition on the classifier)

### 2.3 The Impossibility Theorem

**Chouldechova (2017) and Kleinberg et al. (2017)** proved that demographic parity, equalized odds, and predictive parity cannot all be simultaneously satisfied when base rates differ across groups — except in degenerate cases (perfect classifier or equal base rates).

This means every fairness intervention involves explicit trade-offs. There is no universally "fair" metric — the appropriate one depends on the deployment context:
- Medical diagnosis: Equalized odds (false negative parity) is critical — missing disease should be equally rare across groups
- Credit lending: Predictive parity ensures rates reflect actual risk, but can perpetuate historical inequalities
- Hiring: Demographic parity may be required by law (disparate impact doctrine)

```python
import numpy as np
from sklearn.metrics import confusion_matrix

def compute_fairness_metrics(y_true, y_pred, protected_attribute):
    """Compute key fairness metrics for a binary protected attribute."""
    results = {}
    for group_val in [0, 1]:
        mask = protected_attribute == group_val
        y_t = y_true[mask]
        y_p = y_pred[mask]
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        n = len(y_t)
        results[group_val] = {
            "positive_rate":    (tp + fp) / n,
            "tpr":              tp / (tp + fn + 1e-9),
            "fpr":              fp / (fp + tn + 1e-9),
            "precision":        tp / (tp + fp + 1e-9),
        }

    g0, g1 = results[0], results[1]
    print("=== Fairness Metrics ===")
    print(f"Demographic Parity Diff:  {abs(g0['positive_rate'] - g1['positive_rate']):.4f}")
    print(f"Equalized Odds (TPR diff): {abs(g0['tpr'] - g1['tpr']):.4f}")
    print(f"Equalized Odds (FPR diff): {abs(g0['fpr'] - g1['fpr']):.4f}")
    print(f"Predictive Parity Diff:    {abs(g0['precision'] - g1['precision']):.4f}")
    return results
```

### 2.4 Bias Mitigation Strategies

**Pre-processing (data level):**
- **Reweighting:** Assign higher sample weights to underrepresented groups
- **Resampling:** Oversample minority groups or undersample majority groups
- **Data augmentation:** Generate synthetic examples for underrepresented groups
- **Disparate impact remover:** Transform feature distributions to remove correlation with protected attributes while preserving rank order

**In-processing (model level):**
- **Fairness constraints:** Add regularization terms that penalize fairness metric violations
- **Adversarial debiasing:** Train a secondary "adversary" model that tries to predict protected attributes from model representations; primary model is trained to confuse it

  ```python
  # Adversarial debiasing sketch
  # Main model loss + λ × adversary confusion loss
  L_total = L_task - λ × L_adversary(predict_protected_attr from representation)
  ```

- **Reductions approach (Agarwal et al., 2018):** Reformulate fairness constraints as a cost-sensitive classification problem; available in Fairlearn

**Post-processing (prediction level):**
- **Threshold optimization:** Set different classification thresholds per group to satisfy a fairness constraint
- **Calibrated equalized odds (Pleiss et al., 2017):** Adjust scores post-hoc to equalize odds across groups

```python
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import MetricFrame, demographic_parity_difference

optimizer = ThresholdOptimizer(
    estimator=base_model,
    constraints="demographic_parity",
    objective="balanced_accuracy_score",
    predict_method="predict_proba"
)
optimizer.fit(X_train, y_train, sensitive_features=A_train)
y_pred_fair = optimizer.predict(X_test, sensitive_features=A_test)
```

**Auditing tools:**
- **Fairlearn** — Microsoft's open-source toolkit: [fairlearn.org](https://fairlearn.org)
- **AI Fairness 360 (AIF360)** — IBM's comprehensive toolkit: [aif360.mybluemix.net](https://aif360.mybluemix.net)
- **What-If Tool** — Google's interactive visualization for model fairness

---

## 3. Explainability and Interpretability (XAI)

### 3.1 The Explainability Landscape

| Dimension | Description | Examples |
|---|---|---|
| **Scope** | Global (model-level) vs. Local (instance-level) | Feature importance vs. single prediction explanation |
| **Approach** | Intrinsic (interpretable by design) vs. Post-hoc (applied after training) | Decision tree vs. SHAP on a neural network |
| **Method** | Model-agnostic vs. Model-specific | LIME (any model) vs. Gradient-based (differentiable only) |

### 3.2 SHAP (SHapley Additive exPlanations)

SHAP (Lundberg & Lee, 2017) assigns each feature a Shapley value — its average marginal contribution to predictions across all possible feature subsets. This provides a theoretically grounded, consistent attribution.

**SHAP values are additive:** For prediction `f(x)`:

```
f(x) = E[f(x)] + Σ_i φ_i
```

where `φ_i` is the Shapley value for feature `i` and `E[f(x)]` is the baseline prediction.

```python
import shap
import xgboost as xgb

# Train a model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Visualizations
shap.plots.waterfall(shap_values[0])        # Single prediction breakdown
shap.plots.beeswarm(shap_values)             # Global feature importance
shap.plots.scatter(shap_values[:, "age"])    # Feature interaction plot

# Summary: mean absolute SHAP value = global feature importance
shap.plots.bar(shap_values)
```

**TreeSHAP:** Exact SHAP computation for tree-based models in polynomial time (O(TLD²) where T=trees, L=leaves, D=depth). For neural networks, **DeepSHAP** or **KernelSHAP** (model-agnostic, slower) are alternatives.

### 3.3 LIME (Local Interpretable Model-Agnostic Explanations)

LIME (Ribeiro et al., 2016) explains a single prediction by training an interpretable surrogate model (linear regression or decision tree) on a locally perturbed neighborhood around the input.

```
ξ(x) = argmin_{g ∈ G} L(f, g, π_x) + Ω(g)
```

where `f` is the black-box model, `g` is the interpretable surrogate, `π_x` weights samples by distance to `x`, and `Ω(g)` penalizes model complexity.

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=["No Default", "Default"],
    mode="classification"
)

explanation = explainer.explain_instance(
    X_test[0],
    model.predict_proba,
    num_features=10
)
explanation.show_in_notebook()
```

**LIME limitations:**
- Explanations can be unstable — perturbing `x` slightly can yield different explanations
- Local linear approximation may be poor in highly nonlinear regions

### 3.4 Gradient-Based Explanations

For differentiable models, gradients provide a natural measure of input sensitivity.

**Vanilla gradients:** `∂f(x) / ∂x_i` — how much does the output change with respect to each input feature?

**Integrated Gradients (Sundararajan et al., 2017):** Accumulates gradients along a path from a baseline `x'` to the input `x`:

```
IntegratedGrad_i(x) = (x_i - x'_i) × ∫₀¹ (∂F(x' + α(x-x')) / ∂x_i) dα
```

Satisfies the **completeness axiom**: attributions sum to `F(x) - F(x')`.

```python
import torch

def integrated_gradients(model, input_tensor, baseline, steps=50):
    input_tensor.requires_grad_(True)
    path = [baseline + (float(i) / steps) * (input_tensor - baseline)
            for i in range(steps + 1)]
    grads = []
    for alpha_input in path:
        alpha_input = alpha_input.requires_grad_(True)
        output = model(alpha_input)
        model.zero_grad()
        output.backward(torch.ones_like(output))
        grads.append(alpha_input.grad.detach())

    avg_grads = torch.stack(grads).mean(dim=0)
    attributions = (input_tensor - baseline) * avg_grads
    return attributions
```

### 3.5 Limitations of Post-hoc Explainability

- **Faithfulness gap:** Explanations describe a surrogate model, not necessarily the actual model's decision process.
- **Manipulation risk:** Models can be constructed to game SHAP/LIME explanations — producing "fair-looking" explanations for discriminatory decisions.
- **Human interpretability ≠ correctness:** An explanation that looks plausible may still misrepresent the actual reasoning.
- **Attention ≠ explanation:** Attention weights in transformers are not direct evidence of what the model "looked at" for its decision (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019).

---

## 4. Privacy-Preserving Machine Learning

### 4.1 Privacy Risks in ML

Training data can leak through model outputs:

- **Membership inference attacks:** Determine whether a specific example was in the training set (exploits generalization gap between train and test loss)
- **Model inversion attacks:** Reconstruct training examples from model parameters or predictions
- **Attribute inference:** Infer sensitive attributes of training examples from model outputs
- **Gradient leakage:** In federated learning, shared gradients can reveal local training data (Zhu et al., 2019)

### 4.2 Differential Privacy

Differential Privacy (DP) provides a rigorous mathematical guarantee: an algorithm's output distribution is nearly identical whether or not any single individual's data is included.

**Formal definition (ε-DP):**

A randomized algorithm `M` satisfies ε-differential privacy if for all datasets `D₁` and `D₂` differing in one element, and for all outputs `S ⊆ Range(M)`:

```
P[M(D₁) ∈ S] ≤ e^ε × P[M(D₂) ∈ S]
```

Smaller `ε` = stronger privacy. Common deployments: `ε ∈ [1, 10]` for ML applications.

**DP-SGD (Abadi et al., 2016):** Adds calibrated Gaussian noise to per-sample gradients during training.

```python
from opacus import PrivacyEngine
import torch.optim as optim

model = MyModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

privacy_engine = PrivacyEngine()
model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    target_epsilon=5.0,      # Privacy budget
    target_delta=1e-5,       # Probability of privacy failure
    max_grad_norm=1.0,       # Gradient clipping bound
    epochs=10,
)

# Training loop is identical to standard PyTorch
for batch in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(batch["x"]), batch["y"])
    loss.backward()
    optimizer.step()

epsilon = privacy_engine.get_epsilon(delta=1e-5)
print(f"Trained with ε={epsilon:.2f}, δ=1e-5")
```

**Privacy-utility trade-off:** DP training typically reduces model accuracy. The degradation depends on dataset size (larger datasets tolerate more noise), `ε` value, and model architecture.

### 4.3 Federated Learning

Federated Learning (McMahan et al., 2017) trains a global model across distributed data holders (e.g., mobile devices, hospitals) without centralizing raw data.

**FedAvg algorithm:**

```
Server:
  1. Initialize global model w_0
  2. For each round t:
     a. Sample subset of clients S_t
     b. Broadcast w_t to clients in S_t
     c. Receive local updates Δw_k from each client k
     d. Aggregate: w_{t+1} = w_t + η × (1/|S_t|) Σ_k Δw_k

Client k (given w_t):
  1. Initialize local model from w_t
  2. Run E epochs of SGD on local data D_k
  3. Send gradient update Δw_k = w_local - w_t to server
```

```python
import copy

def federated_averaging(global_model, client_models, client_data_sizes):
    """FedAvg: aggregate client model updates weighted by dataset size."""
    total_size = sum(client_data_sizes)
    global_state = copy.deepcopy(global_model.state_dict())

    for key in global_state:
        # Weighted average of client parameters
        global_state[key] = sum(
            client_models[i].state_dict()[key] * (client_data_sizes[i] / total_size)
            for i in range(len(client_models))
        )
    global_model.load_state_dict(global_state)
    return global_model
```

**Challenges:**
- **Non-IID data:** Client data distributions differ; standard FedAvg can diverge
- **Stragglers:** Slow clients delay synchronous aggregation
- **Communication cost:** Transmitting model updates is expensive for large models
- **Gradient leakage:** Even gradients can reveal training data — combine with DP for strong guarantees

---

## 5. AI Safety and Alignment

### 5.1 The Alignment Problem

AI alignment is the challenge of ensuring AI systems reliably pursue the goals and values their designers intend, even as systems become more capable.

**The problem is not hypothetical:** Current systems already exhibit alignment failures:

- **Reward hacking / specification gaming:** An agent achieves a high reward score by exploiting gaps between the reward specification and the designer's intent. A cleaning robot instructed to minimize visible mess might cover its camera. A content recommendation system optimizing for engagement may promote outrage.

- **Distributional shift:** A model trained on historical data behaves differently when deployment conditions change. A self-driving system trained on sunny California roads may fail in Finnish winter.

- **Goal misgeneralization:** A model learns a proxy goal that correlates with the true goal during training but diverges in deployment. A model trained to be helpful in English may not generalize helpfulness correctly to other languages.

### 5.2 Capability vs. Alignment Research

| Research Area | Focus | Representative Work |
|---|---|---|
| **Scalable oversight** | Supervising agents on tasks beyond human evaluation capability | Debate (Irving et al.), Recursive reward modeling |
| **Interpretability** | Understanding internal model computations | Mechanistic interpretability (Anthropic), sparse autoencoders |
| **Robustness** | Reliable behavior under distribution shift and adversarial inputs | Certified robustness, adversarial training |
| **Constitutional AI** | Self-improvement via AI-generated principles | Bai et al. (Anthropic, 2022) |
| **Formal verification** | Proving properties about model behavior | Satisfiability and abstraction techniques |
| **Red-teaming** | Probing for failure modes systematically | Automated red-teaming, structured adversarial evaluation |

### 5.3 Adversarial Attacks and Robustness

Adversarial examples are inputs crafted to cause model failures, sometimes imperceptible to humans.

**FGSM (Fast Gradient Sign Method):**

```
x_adv = x + ε × sign(∇_x L(f(x), y))
```

```python
def fgsm_attack(model, x, y, epsilon=0.01):
    """Generate FGSM adversarial example."""
    x.requires_grad_(True)
    output = model(x)
    loss = criterion(output, y)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * x.grad.data.sign()
    x_adv = x + perturbation
    return x_adv.detach()
```

**Adversarial training (Madry et al., 2017):** Train the model on adversarial examples alongside clean examples. The most reliable empirical defense, though it reduces clean accuracy.

**For LLMs — jailbreaks and prompt injection:** Current LLMs are vulnerable to crafted prompts that cause the model to ignore its system prompt or safety constraints. Red-teaming and constitutional AI training are the primary defenses.

### 5.4 Dual-Use Risks and Misuse

Generative AI capabilities have dual-use implications:

| Capability | Beneficial Use | Dual-Use Risk |
|---|---|---|
| Realistic text generation | Writing assistance, translation | Disinformation, phishing, social engineering |
| Image/video generation | Creative tools, film production | Deepfakes, non-consensual intimate imagery |
| Code generation | Developer productivity | Malware development, vulnerability exploitation |
| Scientific knowledge | Research acceleration | Biosecurity risks, chemical weapon synthesis |

**Responsible disclosure:** When security researchers discover critical vulnerabilities in AI systems, coordinated disclosure (informing the developer before public release) allows time for mitigation.

**Risk assessment frameworks:**
- Pre-deployment red-teaming for misuse potential
- Structured access controls (API restrictions, monitoring)
- Content policies with enforced filtering
- Capability restrictions (e.g., refusing to generate certain categories of content)

---

## 6. Responsible AI Development Frameworks

### 6.1 Organizational Frameworks

| Organization | Framework | Key Principles |
|---|---|---|
| **Anthropic** | Constitutional AI | Harmlessness, helpfulness, honesty; AI self-critique via a constitution |
| **Google DeepMind** | Responsible AI Practices | Avoiding unjust impacts; maintaining privacy; testing for safety |
| **Microsoft** | Responsible AI Standard | Fairness, reliability, privacy, inclusiveness, transparency, accountability |
| **IEEE** | Ethically Aligned Design | Human well-being, data agency, effectiveness, transparency, accountability |

### 6.2 Model Cards

Model cards (Mitchell et al., 2019) are structured documentation accompanying ML models that report intended use cases, evaluation results (including fairness metrics), and limitations.

**Essential sections:**
1. **Model Details:** Architecture, training data, version
2. **Intended Use:** Primary use case, out-of-scope uses
3. **Factors:** Relevant groups (demographic, environmental)
4. **Metrics:** Performance measures reported, with disaggregation by factor
5. **Evaluation Data:** Which datasets were used
6. **Ethical Considerations:** Data rights, sensitive uses, bias sources
7. **Caveats and Recommendations:** Known failure modes, maintenance

### 6.3 Datasheets for Datasets

Datasheets (Gebru et al., 2018) provide equivalent documentation for datasets, covering:
- **Motivation:** Why was the dataset created? Who funded it?
- **Composition:** What does it contain? Is there sensitive data?
- **Collection process:** How was data gathered? Was consent obtained?
- **Preprocessing:** What cleaning was applied?
- **Distribution:** Under what license? Where is it hosted?
- **Maintenance:** Who maintains it? Is there a feedback mechanism?

---

## 7. Regulatory and Legal Landscape

### 7.1 EU AI Act (2024)

The EU AI Act establishes a risk-based classification system:

| Risk Tier | Description | Requirements |
|---|---|---|
| **Unacceptable** | Social scoring, real-time biometric surveillance in public spaces | Prohibited |
| **High** | Critical infrastructure, hiring, education, law enforcement | Conformity assessment, human oversight, data governance, logging |
| **Limited** | Chatbots, deepfakes | Transparency obligations (users must know they interact with AI) |
| **Minimal** | Spam filters, AI games | No specific requirements |

**General-purpose AI (GPAI) models** with systemic risk (>10²⁵ FLOPs training compute) face additional obligations: red-teaming, incident reporting, and adversarial testing.

### 7.2 GDPR and the Right to Explanation

Under GDPR Article 22, data subjects have the right not to be subject to solely automated decisions that significantly affect them. They may request meaningful information about the logic involved (a "right to explanation").

This creates a legal requirement for explainability in specific deployment contexts — motivating the XAI techniques covered in Section 3.

### 7.3 US Executive Order on AI (October 2023)

The Biden Administration's EO on AI directed:
- **NIST AI RMF:** Develop standards for AI risk assessment
- **Watermarking:** Guidance on content provenance for AI-generated media
- **Safety evaluations:** Federal agencies to assess AI risks before deployment
- **Algorithmic discrimination:** Enforcement guidance across civil rights laws

---

## 8. Environmental Impact of AI

### 8.1 The Carbon Cost of Training

Large model training consumes significant energy. Key metrics:

```
CO₂_equivalent = Energy (kWh) × Carbon_intensity (kgCO₂/kWh)
```

**Illustrative benchmarks (Strubell et al., 2019; Patterson et al., 2022):**

| Model/Task | CO₂ Equivalent |
|---|---|
| Round-trip trans-Atlantic flight | ~0.9 tCO₂e |
| GPT-3 training | ~502 tCO₂e (estimated) |
| Neural architecture search | ~284 tCO₂e |
| Training a standard BERT base | ~0.65 tCO₂e |

**Note:** These figures vary significantly depending on the energy grid used. Training on renewables-powered infrastructure can reduce CO₂e by 10–100×.

### 8.2 Measuring Compute Efficiency

```python
import time
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(project_name="model_training", output_dir="./emissions")
tracker.start()

# ... model training code ...

emissions = tracker.stop()
print(f"Emissions: {emissions:.4f} kg CO₂")
```

### 8.3 Efficient AI Practices

**Algorithmic efficiency:**
- Prefer efficient architectures (MobileNet, DistilBERT, Mistral) when task permits
- Use PEFT (LoRA, adapters) instead of full fine-tuning
- Apply quantization and pruning at inference time
- Cache repeated computations (KV cache)

**Systems efficiency:**
- Train on hardware with high FLOPs/watt (H100, TPU)
- Use data centers with high Power Usage Effectiveness (PUE < 1.2)
- Prefer renewable energy sources (choose cloud regions accordingly)
- Early stopping: stop training when validation loss plateaus

**Reporting:**
- Report training compute in FLOPs alongside carbon emissions
- Disclose hardware type, training duration, and data center location
- Use efficiency metrics: FLOPs per parameter, performance per kWh

---

## Further Reading

- **Fairness and Machine Learning** — Barocas, S., Hardt, M., Narayanan, A. (2023). *Fairness and Machine Learning: Limitations and Opportunities.* [fairmlbook.org](https://fairmlbook.org)
- **SHAP** — Lundberg, S. & Lee, S. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017. [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)
- **Integrated Gradients** — Sundararajan, M. et al. (2017). *Axiomatic Attribution for Deep Networks.* ICML 2017. [arXiv:1703.01365](https://arxiv.org/abs/1703.01365)
- **DP-SGD** — Abadi, M. et al. (2016). *Deep Learning with Differential Privacy.* CCS 2016. [arXiv:1607.00133](https://arxiv.org/abs/1607.00133)
- **Federated Learning** — McMahan, B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017. [arXiv:1602.05629](https://arxiv.org/abs/1602.05629)
- **Constitutional AI** — Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
- **Model Cards** — Mitchell, M. et al. (2019). *Model Cards for Model Reporting.* FAccT 2019. [arXiv:1810.03993](https://arxiv.org/abs/1810.03993)
- **Datasheets for Datasets** — Gebru, T. et al. (2018). *Datasheets for Datasets.* [arXiv:1803.09010](https://arxiv.org/abs/1803.09010)
- **EU AI Act** — [eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689)
- **Carbon emissions of AI** — Patterson, D. et al. (2022). *The Carbon Footprint of Machine Learning Training.* [arXiv:2104.10350](https://arxiv.org/abs/2104.10350)
- **Fairlearn** — [fairlearn.org](https://fairlearn.org)
- **AI Fairness 360** — [aif360.mybluemix.net](https://aif360.mybluemix.net)
- **Opacus (DP training)** — [opacus.ai](https://opacus.ai)

---

*Last updated: February 2026*
