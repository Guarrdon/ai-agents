# Neural Networks

> 📍 [Home](../../README.md) › [Advanced](../README.md) › Neural Networks

---

## Learning Objectives

By the end of this document you will be able to:

- Describe the mathematical structure of a feedforward neural network
- Explain how backpropagation computes gradients via the chain rule
- Compare common activation functions and understand when to use each
- Identify the role of regularisation techniques in preventing overfitting
- Implement a basic neural network from scratch using PyTorch

---

## Prerequisites

| Topic | Where to Review |
|-------|----------------|
| Matrix multiplication | Any introductory linear algebra resource |
| Derivatives and the chain rule | Calculus refresher (e.g., 3Blue1Brown's *Essence of Calculus*) |
| Python basics | Official Python tutorial |
| What AI is conceptually | [Beginner: What Is AI?](../../beginner/docs/what-is-ai.md) |

---

## 1. The Perceptron: Building Block of Neural Networks

A single **perceptron** computes a weighted sum of its inputs and applies a non-linear activation function:

```
output = activation(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

Where:
- `x₁ ... xₙ` are input features
- `w₁ ... wₙ` are learnable weights
- `b` is the bias term
- `activation` is a non-linear function (e.g., sigmoid, ReLU)

In matrix notation: `output = activation(Wx + b)`

```python
import torch
import torch.nn as nn

# A single linear layer (perceptron without activation)
layer = nn.Linear(in_features=4, out_features=1)

# Forward pass
x = torch.randn(1, 4)  # batch_size=1, input_dim=4
out = layer(x)          # shape: (1, 1)
```

---

## 2. Feedforward Networks (MLPs)

A **Multi-Layer Perceptron (MLP)** stacks multiple perceptron layers, creating a hierarchy of representations:

```
Input → Hidden Layer 1 → Hidden Layer 2 → ... → Output
```

```python
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

The universal approximation theorem guarantees that a single hidden layer with enough neurons can approximate any continuous function — but in practice, depth is far more efficient than width.

---

## 3. Activation Functions

Non-linearity is what gives neural networks their expressive power. Without activation functions, stacked linear layers collapse into a single linear transformation.

| Activation | Formula | Range | Common Use |
|-----------|---------|-------|------------|
| **Sigmoid** | `1 / (1 + e^{-x})` | (0, 1) | Binary output, gates in LSTMs |
| **Tanh** | `(e^x - e^{-x}) / (e^x + e^{-x})` | (-1, 1) | RNNs, normalised range |
| **ReLU** | `max(0, x)` | [0, ∞) | Hidden layers (default choice) |
| **Leaky ReLU** | `max(αx, x)` | (-∞, ∞) | Avoids dying ReLU problem |
| **GELU** | `x · Φ(x)` | (-∞, ∞) | Transformers (BERT, GPT) |
| **SiLU/Swish** | `x · sigmoid(x)` | (-∞, ∞) | Modern LLMs (LLaMA, Mistral) |

> **Key insight:** ReLU's simplicity and non-saturating gradient made it the dominant choice for years. Modern LLMs largely use SiLU (also called Swish) or GELU for their smoother gradient properties.

---

## 4. Backpropagation

Backpropagation applies the **chain rule of calculus** to efficiently compute gradients of the loss with respect to every weight in the network.

### The Chain Rule

For a composed function `L(f(g(x)))`:

```
dL/dx = (dL/df) · (df/dg) · (dg/dx)
```

### Forward and Backward Passes

```python
# Forward pass: compute predictions and loss
model = MLP(input_dim=10, hidden_dim=64, output_dim=1)
criterion = nn.MSELoss()

x = torch.randn(32, 10)   # batch of 32 samples
y = torch.randn(32, 1)    # targets

predictions = model(x)
loss = criterion(predictions, y)

# Backward pass: compute gradients
loss.backward()  # PyTorch auto-differentiates through the graph

# Each parameter now has a .grad attribute
for name, param in model.named_parameters():
    print(f"{name}: grad shape = {param.grad.shape}")
```

PyTorch builds a **dynamic computation graph** during the forward pass, then traverses it in reverse to compute gradients — this is called **autograd**.

---

## 5. Gradient Descent and Optimisers

Once gradients are computed, an **optimiser** updates the weights to minimise the loss.

### Vanilla Gradient Descent

```
θ ← θ - η · ∇L(θ)
```

Where `η` is the learning rate and `∇L(θ)` is the gradient of the loss.

### Adam Optimiser (most common default)

Adam maintains per-parameter first and second moment estimates, providing adaptive learning rates:

```python
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Training step
optimiser.zero_grad()   # Clear previous gradients
loss = criterion(model(x), y)
loss.backward()
optimiser.step()        # Update weights
```

| Optimiser | Adaptive LR | Momentum | Notes |
|-----------|------------|---------|-------|
| SGD | No | Optional | Best final performance with tuning |
| RMSProp | Yes | No | Good for RNNs |
| Adam | Yes | Yes | Safe default; may generalise worse |
| AdamW | Yes | Yes | Adam + decoupled weight decay (recommended) |

---

## 6. Loss Functions

The loss function measures how wrong the model's predictions are. The choice depends on the task:

| Task | Loss Function | PyTorch |
|------|-------------|---------|
| Binary classification | Binary cross-entropy | `nn.BCEWithLogitsLoss()` |
| Multi-class classification | Cross-entropy | `nn.CrossEntropyLoss()` |
| Regression | Mean squared error | `nn.MSELoss()` |
| Regression (robust) | Huber loss | `nn.HuberLoss()` |
| Next-token prediction | Cross-entropy over vocabulary | `nn.CrossEntropyLoss()` |

---

## 7. Regularisation

Regularisation techniques prevent overfitting by constraining the model's complexity.

### Dropout

Randomly zeroes activations during training, forcing the network to learn redundant representations:

```python
class RegularisedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )
```

### Batch Normalisation

Normalises activations across the batch dimension, stabilising training:

```python
nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.BatchNorm1d(hidden_dim),
    nn.ReLU(),
)
```

### Weight Decay (L2 Regularisation)

Penalises large weights, built into the `AdamW` optimiser via the `weight_decay` parameter.

---

## 8. Hyperparameter Tuning

Hyperparameters are configuration choices not learned from data: learning rate, batch size, number of layers, neurons per layer, dropout rate, etc.

### Key Hyperparameters and Sensible Defaults

| Hyperparameter | Typical Range | Sensitivity |
|---|---|---|
| **Learning rate** | `1e-5` to `1e-1` | Very high — most critical |
| **Batch size** | 32–512 | Medium; scale LR linearly with batch size |
| **Number of hidden layers** | 2–6 (tabular), 50–1000 (ResNets) | High |
| **Neurons per layer** | 64–2048 | Medium |
| **Dropout rate** | 0.1–0.5 | Medium |
| **Weight decay** | `1e-5` to `1e-2` | Low |

### Search Strategies

**Random search** is often the most practical starting point — sample hyperparameter values randomly from defined distributions:

```python
import numpy as np

search_space = {
    "lr":         lambda: 10 ** np.random.uniform(-5, -1),
    "batch_size": lambda: int(2 ** np.random.randint(5, 10)),  # 32–512
    "hidden_dim": lambda: int(2 ** np.random.randint(6, 11)),  # 64–1024
    "dropout":    lambda: np.random.uniform(0.1, 0.5),
}
configs = [{k: v() for k, v in search_space.items()} for _ in range(20)]
```

**Bayesian optimisation** (via [Optuna](https://optuna.org)) builds a probabilistic model of the hyperparameter-performance mapping and picks the next configuration more intelligently:

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512, 1024])
    model = build_model(hidden_dim=hidden_dim, dropout=dropout)
    return train_and_evaluate(model, lr=lr, epochs=10)  # Returns validation accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(f"Best params: {study.best_trial.params}")
```

### Validation Strategy

- Use a **held-out validation set** for hyperparameter tuning (never the test set)
- Use **k-fold cross-validation** for small datasets
- Apply **early stopping** based on validation loss to avoid overfitting

---

## 9. Key Concepts Summary

| Concept | One-line description |
|---------|---------------------|
| **Perceptron** | Weighted sum + activation — the atomic unit of a neural network |
| **Backprop** | Chain rule applied to compute gradients through the whole network |
| **Autograd** | PyTorch's automatic differentiation engine |
| **Overfitting** | Model memorises training data and fails to generalise |
| **Regularisation** | Techniques that constrain model complexity to improve generalisation |
| **Learning rate** | Step size for gradient descent; most important hyperparameter |

---

## 🧠 Knowledge Check

Test your understanding before moving on. Attempt each question before revealing the answer.

**Q1.** Without activation functions, what does a 10-layer neural network mathematically reduce to?

<details>
<summary>Answer</summary>

It reduces to a **single linear transformation**. Stacking linear layers — `W₂(W₁x + b₁) + b₂` — is equivalent to `(W₂W₁)x + (W₂b₁ + b₂)`, which is still just `Wx + b`. The composition of linear functions is always linear, so depth provides no additional expressiveness without non-linear activations.

</details>

---

**Q2.** Consider this training loop. What critical step is missing?

```python
for x, y in dataloader:
    predictions = model(x)
    loss = criterion(predictions, y)
    loss.backward()
    optimiser.step()
```

<details>
<summary>Answer</summary>

**`optimiser.zero_grad()` is missing** before `loss.backward()`.

PyTorch accumulates gradients by default. Without clearing them each iteration, gradients from previous batches are summed into the current gradients, producing incorrect weight updates.

```python
for x, y in dataloader:
    optimiser.zero_grad()   # ← clear accumulated gradients
    predictions = model(x)
    loss = criterion(predictions, y)
    loss.backward()
    optimiser.step()
```

</details>

---

**Q3.** Your model achieves 99% training accuracy but only 65% validation accuracy. Name two regularisation techniques you could apply and briefly explain the mechanism behind each.

<details>
<summary>Answer</summary>

The model is **overfitting**. Two interventions:

1. **Dropout** — randomly zeroes a fraction of activations during training (e.g., `nn.Dropout(p=0.3)`). Forces the network to learn distributed, redundant representations rather than relying on any single pathway. Disabled automatically during `model.eval()`.

2. **Weight decay (L2 regularisation)** — penalises large weight magnitudes, discouraging the model from memorising training noise. Applied via the `weight_decay` parameter in AdamW: `AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)`.

Other valid answers: reduce model capacity, add more training data, data augmentation, early stopping.

</details>

---

➡️ **Full quiz with 5 questions:** [Knowledge Checks → Neural Networks](knowledge-checks.md#1-neural-networks)

---

## Further Reading

| Resource | Type | Notes |
|---------|------|-------|
| [Deep Learning — Goodfellow, Bengio & Courville](https://www.deeplearningbook.org/) | Textbook | Chapters 6–8 cover feedforward networks rigorously |
| [Neural Networks and Deep Learning — Michael Nielsen](http://neuralnetworksanddeeplearning.com/) | Online book | Excellent intuition-first treatment |
| [3Blue1Brown Neural Network Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | Video | Superb visual intuition for backprop |
| [PyTorch Autograd Tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) | Documentation | Official guide to PyTorch's autograd system |
| [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) | Code | Build backprop from scratch in ~100 lines |

---

*Navigation: [← Beginner Section](../../beginner/README.md) · [Advanced Home](../README.md) · [Next: Deep Learning Architectures →](deep-learning-architectures.md)*
