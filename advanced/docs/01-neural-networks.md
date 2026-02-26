> 📍 [Home](../../README.md) › [Advanced](../README.md) › Neural Networks — In Depth

# Neural Networks — In Depth

> **Learning Objectives**
>
> By the end of this document, you will be able to:
> - Describe the mathematical model of an artificial neuron and relate it to its biological inspiration
> - Implement forward propagation through a multi-layer network using matrix operations
> - Explain how backpropagation uses the chain rule to compute gradients efficiently
> - Choose appropriate activation functions and loss functions for a given task
> - Apply gradient descent variants (SGD, Momentum, Adam) and understand their trade-offs
> - Use regularization techniques (L1/L2, Dropout, Batch Normalization) to reduce overfitting

---

## 1. The Artificial Neuron

### 1.1 Biological Inspiration vs. Mathematical Abstraction

Biological neurons receive electrical signals through dendrites, integrate them in the cell body (soma), and fire an output signal along the axon if the aggregate signal exceeds a threshold. Artificial neurons model this behavior with a deliberate simplification: inputs are multiplied by learnable weights, summed, shifted by a bias, and passed through a nonlinear activation function.

**The mathematical neuron (perceptron model):**

Given input vector `x ∈ ℝⁿ`, weight vector `w ∈ ℝⁿ`, scalar bias `b ∈ ℝ`:

```
z = w · x + b = Σᵢ wᵢxᵢ + b        (pre-activation / linear combination)
a = σ(z)                              (post-activation / neuron output)
```

where `σ` is an activation function. The neuron learns by adjusting `w` and `b` through training.

**Intuition:** The weight `wᵢ` controls how much feature `xᵢ` influences the output. A large positive weight means the neuron "excites" in response to that feature; a large negative weight means it "inhibits". The bias `b` shifts the activation threshold — it allows the neuron to fire even when all inputs are zero, giving the model greater expressivity.

### 1.2 A Single Neuron as a Linear Classifier

A single neuron with a sigmoid activation computes:

```
P(y=1 | x) = σ(w · x + b) = 1 / (1 + exp(-(w · x + b)))
```

This is identical to logistic regression. A single neuron can only learn **linearly separable** decision boundaries. The power of neural networks comes from stacking multiple neurons in layers.

---

## 2. Activation Functions

Activation functions introduce nonlinearity — without them, a multi-layer network would collapse to a single linear transformation (because the composition of linear functions is linear). The choice of activation function significantly affects training dynamics and performance.

### 2.1 Sigmoid

```
σ(z) = 1 / (1 + exp(-z))        range: (0, 1)
σ'(z) = σ(z)(1 - σ(z))
```

**Properties:**
- Smooth, differentiable everywhere
- Maps any real input to (0, 1) — useful for binary output probabilities
- **Vanishing gradients:** For large |z|, `σ'(z) ≈ 0`. During backpropagation, these near-zero gradients are multiplied together across layers, causing gradient magnitudes to shrink exponentially. Deep networks with sigmoid activations struggle to train.
- Outputs are not zero-centered (always positive), which can slow convergence for gradient descent.

### 2.2 Tanh (Hyperbolic Tangent)

```
tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))        range: (-1, 1)
tanh'(z) = 1 - tanh²(z)
```

**Properties:**
- Zero-centered outputs (unlike sigmoid) — generally preferred over sigmoid for hidden layers
- Stronger gradients near the origin: `tanh'(0) = 1` vs. `σ'(0) = 0.25`
- Still suffers from vanishing gradients for saturated inputs

### 2.3 ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)         range: [0, ∞)
ReLU'(z) = 1 if z > 0, else 0
```

**Properties:**
- Computationally inexpensive (just a max operation)
- No vanishing gradient for positive inputs — gradient is exactly 1
- Sparsity: approximately 50% of neurons output 0 on a given input, making representations sparse and efficient
- **Dying ReLU:** A neuron whose pre-activation `z` is always negative gets zero gradient and never updates. This can occur with a too-large learning rate or poor initialization.

**In practice:** ReLU is the default choice for hidden layers in most feedforward networks and CNNs.

### 2.4 Leaky ReLU and ELU

**Leaky ReLU** addresses the dying ReLU problem by allowing a small negative slope `α ≈ 0.01`:

```
LeakyReLU(z) = z if z > 0, else αz       (typical α = 0.01)
```

**ELU (Exponential Linear Unit):** Smooth for negative inputs, which can speed convergence:

```
ELU(z) = z if z > 0, else α(exp(z) - 1)
```

ELU pushes mean activations closer to zero (like tanh) without sacrificing the positive-domain properties of ReLU.

### 2.5 Softmax (Output Layer for Classification)

For multi-class classification with `K` classes, the softmax function converts raw logits into a probability distribution:

```
softmax(z)ₖ = exp(zₖ) / Σⱼ exp(zⱼ)        Σₖ softmax(z)ₖ = 1
```

Softmax should only be used at the output layer (it normalizes across all output units, so it only makes sense as a group operation).

**Numerical stability:** Computing `exp(z)` can overflow. In practice, subtract the maximum logit first:

```python
def softmax(z):
    z_shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)
```

### 2.6 Activation Function Comparison

| Activation | Range | Zero-centered | Vanishing Gradient | Recommended Use |
|---|---|---|---|---|
| Sigmoid | (0, 1) | No | Yes | Binary output layer |
| Tanh | (-1, 1) | Yes | Yes | RNNs, some hidden layers |
| ReLU | [0, ∞) | No | No (for z>0) | Default hidden layers |
| Leaky ReLU | (-∞, ∞) | No | No | When dying ReLU is a concern |
| ELU | ~(-α, ∞) | Near-zero | No | Regularized models |
| Softmax | (0, 1) per class | N/A | N/A | Multi-class output |

---

## 3. Network Architecture

### 3.1 Layer Types

A **fully connected (dense) layer** connects every neuron in one layer to every neuron in the next. For a layer with `nᵢ` input neurons and `nₒ` output neurons, the layer has `nᵢ × nₒ` weight parameters plus `nₒ` bias parameters.

A network with:
- **Input layer:** `n₀` neurons (one per input feature; no learnable parameters)
- **Hidden layers:** One or more layers of neurons with learnable weights
- **Output layer:** `nL` neurons (one per output class or output dimension)

is called a **feedforward neural network** or **multilayer perceptron (MLP)**.

### 3.2 Universal Approximation Theorem

The Universal Approximation Theorem (Cybenko, 1989; Hornik, 1991) states that a feedforward network with a single hidden layer and a sufficient number of neurons can approximate any continuous function on a compact subset of ℝⁿ to arbitrary precision, given a non-polynomial activation function.

**What this means in practice:**
- The theorem guarantees *existence* of such a network, not that gradient descent will *find* it
- A shallow (1 hidden layer) network may need exponentially many neurons to approximate certain functions that a deeper network can represent compactly
- Depth provides compositional structure that mirrors natural hierarchies in real-world data

### 3.3 Depth vs. Width

| Architecture | Trade-offs |
|---|---|
| **Deeper (more layers)** | More parameter-efficient for hierarchical features; harder to train (vanishing gradients, slower convergence) |
| **Wider (more neurons per layer)** | Easier to optimize; may require many more total parameters |
| **Practical guidance** | Start with moderate depth (3–5 layers for tabular data); use ResNets for very deep vision models |

---

## 4. Forward Propagation

Forward propagation computes the network's prediction given an input. It is a sequence of matrix multiplications and elementwise activations.

### 4.1 Notation

For a network with `L` layers (not counting the input):

- `W^[l] ∈ ℝ^(n^[l] × n^[l-1])` — weight matrix for layer `l`
- `b^[l] ∈ ℝ^(n^[l])` — bias vector for layer `l`
- `Z^[l]` — pre-activation values at layer `l`
- `A^[l]` — post-activation values at layer `l` (with `A^[0] = X`, the input)

### 4.2 Forward Pass Equations

For each layer `l = 1, ..., L`:

```
Z^[l] = W^[l] A^[l-1] + b^[l]        (linear transform)
A^[l] = σ^[l](Z^[l])                  (apply activation)
```

The output prediction is `ŷ = A^[L]`.

### 4.3 NumPy Implementation

```python
import numpy as np

def initialize_parameters(layer_dims: list[int]) -> dict:
    """
    He initialization for ReLU networks.
    layer_dims: [n_input, n_hidden1, ..., n_output]
    """
    params = {}
    L = len(layer_dims) - 1
    for l in range(1, L + 1):
        params[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0 / layer_dims[l-1])
        params[f"b{l}"] = np.zeros((layer_dims[l], 1))
    return params

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def forward_propagation(X: np.ndarray, params: dict, L: int) -> tuple[np.ndarray, dict]:
    """
    Compute forward pass; cache intermediate values for backprop.
    X: (n_input, m) where m is the batch size
    Returns: (predictions, cache)
    """
    cache = {"A0": X}
    A = X

    for l in range(1, L + 1):
        W = params[f"W{l}"]
        b = params[f"b{l}"]
        Z = W @ A + b                           # Linear transform
        A = relu(Z) if l < L else sigmoid(Z)   # ReLU for hidden, sigmoid for output
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A

    return A, cache
```

**Key design note:** Caching `Z^[l]` and `A^[l]` during the forward pass is critical — these values are needed during backpropagation to compute gradients.

---

## 5. Loss Functions

The loss function measures the discrepancy between the network's predictions and the ground truth. Choosing the right loss is determined by the task.

### 5.1 Mean Squared Error (Regression)

```
MSE = (1/m) Σᵢ (ŷᵢ - yᵢ)²
```

**When to use:** Regression tasks (predicting continuous values). Penalizes large errors quadratically, making it sensitive to outliers.

**Huber loss** combines MSE and MAE for outlier-robustness:

```
Lδ(y, ŷ) = 0.5(y - ŷ)²              if |y - ŷ| ≤ δ
           δ|y - ŷ| - 0.5δ²          otherwise
```

### 5.2 Binary Cross-Entropy (Binary Classification)

For a single binary output `ŷ = P(y=1|x)`:

```
BCE = -(1/m) Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

**Intuition:** Cross-entropy measures the information cost of using the predicted distribution to encode the true distribution. Minimizing BCE is equivalent to maximizing the log-likelihood of the data under the model.

**Numerical note:** Use `log(ŷ + ε)` with `ε = 1e-7` to avoid `log(0)`. PyTorch's `BCEWithLogitsLoss` handles this by combining sigmoid + BCE in a numerically stable way.

### 5.3 Categorical Cross-Entropy (Multi-class Classification)

For a softmax output with `K` classes (using one-hot encoded labels `y`):

```
CCE = -(1/m) Σᵢ Σₖ yᵢₖ log(ŷᵢₖ)
```

Since labels are one-hot, only the log-probability of the correct class contributes:

```
CCE = -(1/m) Σᵢ log(ŷᵢ[cᵢ])        where cᵢ is the true class index
```

### 5.4 PyTorch Loss Functions

```python
import torch
import torch.nn as nn

# Regression
mse_loss = nn.MSELoss()

# Binary classification (with logits — numerically preferred)
bce_loss = nn.BCEWithLogitsLoss()

# Multi-class classification (takes raw logits, not softmax outputs)
ce_loss = nn.CrossEntropyLoss()

# Example usage
logits = torch.tensor([[2.1, 0.3, -0.8], [0.1, 1.5, 0.3]])
labels = torch.tensor([0, 1])   # True class indices
loss = ce_loss(logits, labels)
print(f"Cross-entropy loss: {loss.item():.4f}")
```

---

## 6. Backpropagation

Backpropagation (Rumelhart, Hinton & Williams, 1986) is the algorithm that computes gradients of the loss with respect to every parameter in the network. It is an efficient application of the **chain rule of calculus**, traversing the network in reverse.

### 6.1 The Chain Rule

For a composition of functions `L = f(g(h(x)))`:

```
dL/dx = (dL/df) × (df/dg) × (dg/dh) × (dh/dx)
```

In a neural network, we want `∂L/∂W^[l]` and `∂L/∂b^[l]` for each layer `l`. Backpropagation computes these by reusing intermediate gradient values, propagating the **error signal** backwards from the output layer to the input.

### 6.2 Backpropagation Equations

Define the **error term** for layer `l`:

```
δ^[L] = ∂L/∂Z^[L]                                    (output layer)
δ^[l] = (W^[l+1])ᵀ δ^[l+1] ⊙ σ'^[l](Z^[l])          (hidden layers)
```

where `⊙` denotes elementwise multiplication and `σ'^[l]` is the derivative of the activation at layer `l`.

The weight and bias gradients are then:

```
∂L/∂W^[l] = (1/m) δ^[l] (A^[l-1])ᵀ
∂L/∂b^[l] = (1/m) Σᵢ δ^[l]ᵢ              (mean over batch)
```

### 6.3 NumPy Backpropagation Implementation

```python
def relu_backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Gradient of ReLU: pass gradient only where Z > 0."""
    return dA * (Z > 0)

def sigmoid_backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Gradient of sigmoid."""
    s = sigmoid(Z)
    return dA * s * (1 - s)

def compute_loss(A_out: np.ndarray, Y: np.ndarray) -> float:
    """Binary cross-entropy loss."""
    m = Y.shape[1]
    eps = 1e-7
    return -(1.0 / m) * np.sum(Y * np.log(A_out + eps) + (1 - Y) * np.log(1 - A_out + eps))

def backward_propagation(Y: np.ndarray, cache: dict, params: dict, L: int) -> dict:
    """
    Compute gradients for all parameters.
    Y: (1, m) true labels
    Returns: dict of gradients dW1, db1, ..., dWL, dbL
    """
    m = Y.shape[1]
    grads = {}

    # Output layer gradient (binary cross-entropy + sigmoid)
    A_out = cache[f"A{L}"]
    dA = -(Y / (A_out + 1e-7)) + ((1 - Y) / (1 - A_out + 1e-7))

    for l in reversed(range(1, L + 1)):
        Z = cache[f"Z{l}"]
        A_prev = cache[f"A{l-1}"]

        # Gradient through activation
        dZ = sigmoid_backward(dA, Z) if l == L else relu_backward(dA, Z)

        # Gradient for weights and biases
        grads[f"dW{l}"] = (1 / m) * dZ @ A_prev.T
        grads[f"db{l}"] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Gradient to pass to previous layer
        dA = params[f"W{l}"].T @ dZ

    return grads
```

### 6.4 Visualizing Gradient Flow

Consider a 3-layer network. The gradient of the loss at the output must pass through every layer to reach the first layer's weights:

```
∂L/∂W¹ ∝ δ¹ = (W²)ᵀ δ² ⊙ σ'(Z¹)
                    ↑
         (W³)ᵀ δ³ ⊙ σ'(Z²)
                    ↑
         ∂L/∂Z³  (output gradient)
```

Each step involves multiplying by `W^[l+1]` and `σ'(Z^[l])`. If activation derivatives are consistently small (sigmoid) or weight matrices have singular values < 1, the gradient **vanishes** — becoming exponentially small as it travels back through layers. Conversely, if weight values or activation gradients are large, the gradient may **explode**.

---

## 7. Optimization Algorithms

Optimization algorithms update parameters using gradients to minimize the loss. The choice of optimizer significantly affects convergence speed and final model quality.

### 7.1 Gradient Descent Variants

**Batch gradient descent** computes the gradient over the entire dataset — expensive for large datasets, but produces stable gradient estimates.

**Stochastic Gradient Descent (SGD)** updates parameters using the gradient from a single randomly selected training example:

```
θ ← θ - α ∇_θ L(θ; xᵢ, yᵢ)
```

Noisy gradients allow SGD to escape shallow local minima and saddle points, but cause erratic convergence.

**Mini-batch SGD** (the dominant approach) computes gradients over a small batch of `B` examples, balancing noise and efficiency:

```
θ ← θ - α ∇_θ L(θ; X_batch, Y_batch)
```

Typical batch sizes: 32–512 for standard training; larger batches benefit from linear scaling of the learning rate.

### 7.2 Momentum

Standard SGD oscillates in directions orthogonal to the gradient minimum. Momentum accumulates a velocity vector `v` that dampens oscillations and accelerates movement in consistent directions:

```
v ← βv + (1-β) ∇_θ L(θ)          (momentum)
θ ← θ - α v
```

Common setting: `β = 0.9`. With `β=0`, this reduces to standard SGD.

### 7.3 RMSProp

RMSProp (Hinton, 2012) adapts the learning rate per-parameter by dividing by the root mean square of recent gradients. This prevents large updates for parameters with large gradients:

```
S ← βS + (1-β) (∇_θ L)²           (exponential moving average of squared gradients)
θ ← θ - (α / √(S + ε)) ∇_θ L
```

### 7.4 Adam (Adaptive Moment Estimation)

Adam (Kingma & Ba, 2015) combines momentum (first moment) and RMSProp (second moment) with bias correction to handle the zero-initialization of moment vectors:

```
m ← β₁m + (1-β₁) g              (first moment — mean)
v ← β₂v + (1-β₂) g²             (second moment — uncentered variance)

m̂ = m / (1 - β₁ᵗ)               (bias-corrected first moment)
v̂ = v / (1 - β₂ᵗ)               (bias-corrected second moment)

θ ← θ - α m̂ / (√v̂ + ε)
```

**Typical hyperparameters:** `α = 0.001`, `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`.

Adam is the default optimizer for most modern deep learning, particularly for transformers. AdamW (decoupled weight decay) is preferred when L2 regularization is needed.

### 7.5 PyTorch Optimizer Usage

```python
import torch
import torch.optim as optim

model = MyNeuralNet()

# Standard SGD with momentum
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (default choice for most tasks)
optimizer_adam = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

# AdamW (Adam with decoupled weight decay — preferred for transformers)
optimizer_adamw = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Training loop
for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        optimizer_adamw.zero_grad()             # Clear previous gradients
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()                          # Compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer_adamw.step()                   # Update parameters
```

### 7.6 Learning Rate Schedules

A fixed learning rate is rarely optimal. Common schedules:

| Schedule | Behavior | Best For |
|---|---|---|
| **Step decay** | Multiply LR by γ every k epochs | Classic training, image classification |
| **Cosine annealing** | Smooth decay following a cosine curve | LLM fine-tuning, long training runs |
| **Warmup + decay** | Linear ramp-up then decay | Transformers (prevents early instability) |
| **Cyclical LR (Smith, 2017)** | Oscillates between min/max LR | Can accelerate convergence |

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

for epoch in range(100):
    train_one_epoch(model, optimizer, dataloader)
    scheduler.step()
    print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()[0]:.6f}")
```

---

## 8. Regularization

Regularization reduces overfitting by constraining model complexity or adding noise during training. A model that fits training data perfectly but fails on unseen data has **overfit** — regularization improves **generalization**.

### 8.1 L2 Regularization (Weight Decay)

Adds a penalty proportional to the squared magnitude of all weights to the loss:

```
L_regularized = L_original + (λ/2) Σᵢ wᵢ²
```

**Effect:** Discourages large weight values, pushing them toward zero (but not exactly zero). The gradient of the penalty term `λw` is added to each weight's gradient, which is equivalent to decaying weights by a factor `(1 - αλ)` at each step.

**PyTorch:** Supported natively via `weight_decay` in most optimizers.

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

### 8.2 L1 Regularization

Adds a penalty proportional to the absolute weight values:

```
L_regularized = L_original + λ Σᵢ |wᵢ|
```

**Effect:** Promotes **sparsity** — many weights are driven exactly to zero, providing a form of automatic feature selection. L1 is less common in deep learning than L2 but useful for interpretability.

### 8.3 Dropout

Dropout (Srivastava et al., 2014) randomly sets a fraction `p` of neuron activations to zero during training. Each training step uses a different random mask, effectively training an ensemble of exponentially many thinned networks.

```python
import torch.nn as nn

class MLPWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),            # Applied during training only
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

# Dropout is automatically disabled during evaluation
model.train()   # Dropout active
model.eval()    # Dropout disabled; activations scaled by (1-p) implicitly
```

**Inverted dropout (standard implementation):** During training, active neurons are scaled up by `1/(1-p)` so that the expected activation magnitude is the same at test time.

**Practical guidance:**
- Dropout rate `p = 0.5` for large hidden layers is a common starting point
- Lower rates (`p = 0.1–0.3`) for smaller models or convolutional layers
- Do not apply dropout to the output layer

### 8.4 Batch Normalization

Batch Normalization (Ioffe & Szegedy, 2015) normalizes the pre-activations of a layer across the batch dimension during training, then applies a learnable scale `γ` and shift `β`:

```
μ_B = (1/m) Σᵢ zᵢ                          (batch mean)
σ²_B = (1/m) Σᵢ (zᵢ - μ_B)²                (batch variance)
ẑᵢ = (zᵢ - μ_B) / √(σ²_B + ε)              (normalize)
y = γẑ + β                                  (scale and shift — learnable)
```

**Benefits:**
- Reduces **internal covariate shift** — as upstream layer weights change, downstream layers must adapt to a shifting input distribution. BN stabilizes this.
- Acts as a regularizer (adding noise via batch statistics), often reducing the need for Dropout
- Allows much higher learning rates
- Reduces sensitivity to weight initialization

```python
class MLPWithBatchNorm(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),           # After linear, before activation
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
```

**At inference time:** BN uses running statistics accumulated during training (exponential moving average of batch means and variances) rather than batch statistics. This is automatic in PyTorch when using `model.eval()`.

**Layer Normalization** normalizes across the feature dimension rather than the batch dimension — preferred for transformers and RNNs where batch statistics are less stable.

---

## 9. Hyperparameter Tuning

Hyperparameters are configuration choices not learned from data: learning rate, batch size, number of layers, number of neurons per layer, dropout rate, regularization strength, etc.

### 9.1 Key Hyperparameters and Sensible Defaults

| Hyperparameter | Typical Range | Sensitivity |
|---|---|---|
| **Learning rate** | `1e-5` to `1e-1` | Very high — most critical |
| **Batch size** | 32–512 | Medium; scale LR linearly with batch size |
| **Number of hidden layers** | 2–6 (tabular), 50–1000 (ResNets) | High |
| **Neurons per layer** | 64–2048 | Medium |
| **Dropout rate** | 0.1–0.5 | Medium |
| **Weight decay** | `1e-5` to `1e-2` | Low |

### 9.2 Search Strategies

**Grid search:** Evaluate every combination of a predefined set of hyperparameter values. Exhaustive but computationally expensive for many hyperparameters (combinatorial explosion).

**Random search** (Bergstra & Bengio, 2012): Sample hyperparameter values randomly from defined distributions. Often finds good solutions with far fewer evaluations than grid search, especially when some hyperparameters matter much more than others.

```python
import numpy as np
from itertools import product

def random_search(num_trials: int, param_distributions: dict) -> list[dict]:
    """Sample `num_trials` random hyperparameter configurations."""
    configs = []
    for _ in range(num_trials):
        config = {k: v() for k, v in param_distributions.items()}
        configs.append(config)
    return configs

# Define search space using callables
search_space = {
    "lr":          lambda: 10 ** np.random.uniform(-5, -1),
    "batch_size":  lambda: int(2 ** np.random.randint(5, 10)),  # 32 to 512
    "hidden_dim":  lambda: int(2 ** np.random.randint(6, 11)),  # 64 to 1024
    "dropout":     lambda: np.random.uniform(0.1, 0.5),
}

configs = random_search(num_trials=20, param_distributions=search_space)
```

**Bayesian optimization:** Builds a probabilistic surrogate model (e.g., Gaussian process) of the hyperparameter-performance mapping. Uses this model to choose the next configuration to evaluate — balancing exploitation (evaluate near the current best) and exploration (evaluate uncertain regions). More sample-efficient than random search for expensive models.

Libraries: **Optuna** (recommended), **Hyperopt**, **Ray Tune**.

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512, 1024])

    model = build_model(hidden_dim=hidden_dim, dropout=dropout)
    val_accuracy = train_and_evaluate(model, lr=lr, epochs=10)
    return val_accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(f"Best trial: {study.best_trial.params}")
```

### 9.3 Validation Strategy

Always evaluate hyperparameters on a held-out **validation set** — never the test set (which should remain unseen until final evaluation):

- **Random split:** 80/10/10 train/val/test is a common starting point
- **k-Fold cross-validation:** Average performance across `k` folds; preferable for small datasets where a single split may be unrepresentative
- **Early stopping:** Monitor validation loss; stop training when it stops improving to avoid overfitting

---

## Further Reading

- **Neural Networks and Deep Learning** — Nielsen, M. (2015). *Neural Networks and Deep Learning* (free online book). [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com)
- **Deep Learning textbook** — Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press. [deeplearningbook.org](https://www.deeplearningbook.org)
- **Backpropagation** — Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986). *Learning representations by back-propagating errors.* Nature, 323, 533–536.
- **Adam optimizer** — Kingma, D.P. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- **Dropout** — Srivastava, N. et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR 15(1), 1929–1958. [jmlr.org](https://jmlr.org/papers/v15/srivastava14a.html)
- **Batch Normalization** — Ioffe, S. & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training.* ICML 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- **Universal Approximation** — Hornik, K. (1991). *Approximation capabilities of multilayer feedforward networks.* Neural Networks, 4(2), 251–257.
- **Random Search for Hyperparameters** — Bergstra, J. & Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization.* JMLR 13, 281–305.
- **Optuna** — [optuna.org](https://optuna.org) — Hyperparameter optimization framework
- **PyTorch documentation** — [pytorch.org/docs](https://pytorch.org/docs/stable/)

---

*Navigation: [Advanced Home](../README.md) · [Next: Deep Learning Architectures →](02-deep-learning-architectures.md)*

*Last updated: February 2026*
