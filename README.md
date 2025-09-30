# PyTorch-Transformers

# 🧩 Embeddings in PyTorch

In Transformers, the first step is usually converting token IDs into dense vectors using an **embedding layer**. In PyTorch, this is handled by `nn.Embedding`.

---

## 🔹 What is `nn.Embedding`?

Think of `nn.Embedding` as a **lookup table**:

* It stores a trainable weight matrix `W` of shape `[vocab_size, d_model]`.
* Each row in `W` is the embedding vector for one token ID.
* When you pass token IDs as input, the layer simply fetches the corresponding rows.

---

## 🔹 Example Setup

```python
import torch
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=5, embedding_dim=3)
print("Embedding Matrix (W):")
print(embedding.weight)
```

The weight matrix might look like this (random init):
$$
W =
\begin{bmatrix}
w_{0,0} & w_{0,1} & w_{0,2} \\
w_{1,0} & w_{1,1} & w_{1,2} \\
w_{2,0} & w_{2,1} & w_{2,2} \\
w_{3,0} & w_{3,1} & w_{3,2} \\
w_{4,0} & w_{4,1} & w_{4,2}
\end{bmatrix}
$$


---

## 🔹 Case 1: Single Token

```python
x = torch.tensor([1])
out = embedding(x)
```

* Input `x = [1]` → fetch row `1` of `W`.
* Output:
  [
  [w_{1,0}, w_{1,1}, w_{1,2}]
  ]
* Shape:

  * Input = `[1]`
  * Output = `[1, 3]`

✅ A single token ID returns a single embedding vector.

---

## 🔹 Case 2: Batch of Sequences

```python
x = torch.tensor([[1, 2], [3, 4]])
out = embedding(x)
```

* Input shape = `[2, 2]` (2 sequences × 2 tokens each).
* For each token ID in `x`, fetch the corresponding row in `W`:

[
\text{output} =
\begin{bmatrix}
[W[1]] & [W[2]] \
[W[3]] & [W[4]]
\end{bmatrix}
]

* Output shape = `[2, 2, 3]`

✅ Each token ID is replaced by its embedding vector.

---

## 🔹 Visual Intuition

If the embedding matrix is:

```
0 → [0.1, 0.2, 0.3]
1 → [0.4, 0.5, 0.6]
2 → [0.7, 0.8, 0.9]
3 → [1.0, 1.1, 1.2]
4 → [1.3, 1.4, 1.5]
```

* `x = [1]` → `[[0.4, 0.5, 0.6]]`
* `x = [[1, 2], [3, 4]]` →

```
[
  [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
  [[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]]
]
```

---

## 🔹 How Training Updates Embeddings

The interesting part: **only the rows you use get updated**.

### Step 1: Forward Pass

```python
embedding = nn.Embedding(5, 3)
optimizer = torch.optim.SGD(embedding.parameters(), lr=0.1)

x = torch.tensor([1, 3])
out = embedding(x)
```

* `x = [1, 3]` → fetch row `1` and row `3`.

---

### Step 2: Backward Pass

```python
loss = out.sum()
loss.backward()
print(embedding.weight.grad)
```

Gradients:

```
Row 0: [0, 0, 0]
Row 1: [1, 1, 1]
Row 2: [0, 0, 0]
Row 3: [1, 1, 1]
Row 4: [0, 0, 0]
```

👉 Only **rows 1 and 3** (the ones we looked up) have gradients.

---

### Step 3: Update

```python
optimizer.step()
print(embedding.weight.data)
```

* Row 1 and Row 3 get updated.
* Other rows remain unchanged.

---

## 🔹 Why is This Efficient?

* Large vocabularies (e.g. 50k tokens) don’t need to update every row each step.
* Only the embeddings of tokens in the current batch are updated.
* This makes training embeddings scalable and memory-efficient.

---

## ✅ Key Takeaways

* `nn.Embedding` = **row lookup** in a trainable matrix.
* Input IDs → output embedding vectors.
* Output shape = `input_shape + [embedding_dim]`.
* During training, only the rows corresponding to used token IDs are updated.

---

👉 This mechanism allows Transformers to handle massive vocabularies efficiently.

---





# Positional Embeddings in Transformers

In Transformers, **token embeddings** alone do not encode the position of a token in the sequence. To give the model information about token order, we use **positional embeddings**.

Transformers process tokens **in parallel**. Unlike RNNs, they have no notion of sequential order. Positional encoding provides **unique position information** so the model can distinguish between `token_1 token_2` and `token_2 token_1`.

---

## 🔹 Sinusoidal Positional Encoding

The classic method uses **sin and cos functions** of different frequencies:

$$
[
\text{PE}*{(pos, 2i)} = \sin \left(\frac{pos}{10000^{2i/d*\text{model}}} \right)
]

[
\text{PE}*{(pos, 2i+1)} = \cos \left(\frac{pos}{10000^{2i/d*\text{model}}} \right)
]
$$
Where:

* $ pos $ = position index (0,1,...,seq_len-1)
* $ i $ = embedding dimension index (0,...,d_model-1)
* $ d_\text{model} $ = embedding size

✅ This generates **unique vectors for each position**, with smooth variations across dimensions, allowing the model to encode order information.

---

## 🔹 How it’s implemented in code

```python
pe = torch.zeros(seq_len, d_model)            # initialize positional matrix
position = torch.arange(0, seq_len).unsqueeze(1)  # shape: (seq_len, 1)
div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))

pe[:, 0::2] = torch.sin(position * div_term)  # even dims
pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

pe = pe.unsqueeze(0)  # add batch dim -> shape (1, seq_len, d_model)
```

* `unsqueeze(1)` → converts `[seq_len]` → `[seq_len, 1]` for broadcasting.
* `0::2` → even dimensions get `sin`, `1::2` → odd dimensions get `cos`.
* `unsqueeze(0)` → adds **batch dimension** for addition to `(batch_size, seq_len, d_model)` embeddings.

---

## 🔹 Example Positional Encoding

Suppose:

* `seq_len = 5` (5 tokens)
* `d_model = 4` (embedding size)

| Position | Dim 0 (sin) | Dim 1 (cos) | Dim 2 (sin) | Dim 3 (cos) |
| -------- | ----------- | ----------- | ----------- | ----------- |
| 0        | 0.0         | 1.0         | 0.0         | 1.0         |
| 1        | 0.8415      | 0.5403      | 0.0099998   | 0.99995     |
| 2        | 0.9093      | -0.4161     | 0.0199987   | 0.9998      |
| 3        | 0.1411      | -0.9900     | 0.029995    | 0.99955     |
| 4        | -0.7568     | -0.6536     | 0.039989    | 0.9992      |

* **Even dims** → $$sin(pos / 10000^(2i/d_model))$$
* **Odd dims** → $$cos(pos / 10000^(2i/d_model))$$
* Each row represents a **unique positional vector** for a token.

---

## 🔹 Adding Positional Encoding to Token Embeddings

Suppose token embeddings `X` (batch_size=1, seq_len=5, d_model=4) look like:

```
X = [
 [ [0.1, 0.2, 0.3, 0.4],
   [0.5, 0.6, 0.7, 0.8],
   [0.9, 1.0, 1.1, 1.2],
   [1.3, 1.4, 1.5, 1.6],
   [1.7, 1.8, 1.9, 2.0] ]
]
```

After adding positional encoding:

```
X + PE = [
 [ [0.1+0, 0.2+1, 0.3+0, 0.4+1],
   [0.5+0.8415, 0.6+0.5403, 0.7+0.0099998, 0.8+0.99995],
   [0.9+0.9093, 1.0-0.4161, 1.1+0.019998, 1.2+0.9998],
   [1.3+0.1411, 1.4-0.9900, 1.5+0.029995, 1.6+0.99955],
   [1.7-0.7568, 1.8-0.6536, 1.9+0.039989, 2.0+0.9992] ]
]
```

* Each token embedding now carries **both semantic info** (from token embedding) **and position info** (from PE).

---

## 🔹 Why `register_buffer` and `.requires_grad(False)`

```python
self.register_buffer('pe', pe)
x = x + self.pe[:, :x.shape[1], :].requires_grad(False)
```

* **`register_buffer`** → stores `pe` in the model but **not trainable**.
* **`.requires_grad(False)`** → ensures positional encodings **do not get updated** during backprop.
* **Slice `:x.shape[1]`** → matches actual sequence length.
* **Result** → token embeddings + positional embeddings ready for transformer layers.

---


## ✅ Summary

* Positional embeddings allow the model to **encode token order** in a parallel transformer.
* Sinusoidal functions give **smooth, unique patterns** for each position.
* `register_buffer` makes them **non-trainable but GPU-compatible**.
* Forward pass **adds positional encoding** to token embeddings, ready for transformer layers.

---






# Layer Normalization in Transformers

**Layer Normalization (LayerNorm)** is a technique used to **stabilize and accelerate training** by normalizing the inputs to a layer. Unlike Batch Normalization, which normalizes across the batch, LayerNorm normalizes across **features of a single token embedding**.

Transformers process tokens **in parallel**. Each token embedding can have varying magnitudes across dimensions, which can make training unstable. LayerNorm ensures that each token embedding has:

* **Zero mean**
* **Unit variance**

This helps with **faster convergence**, better gradient flow, and improved model stability.

---

## 🔹 Mathematical Formula

Given a token embedding vector ( \mathbf{x} \in \mathbb{R}^{d} ) (with `d_model` features):

1. Compute **mean**:
$
[
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i
]
$
2. Compute **standard deviation**:
$
[
\sigma = \sqrt{\frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2}
]
$
3. Normalize the embedding:
$
[
\hat{x}_i = \frac{x_i - \mu}{\sigma + \epsilon}
]
$
4. Apply **learnable scale and shift**:
$
[
y_i = \alpha \hat{x}_i + \beta
]
$
Where:

* $ \alpha $ = **learnable scale parameter** (`nn.Parameter`)
* $ \beta $ = **learnable shift/bias parameter** (`nn.Parameter`)
* $ \epsilon $ = small constant to prevent division by zero

✅ The result is a normalized token embedding with **controllable magnitude and shift**, allowing the network to adjust normalization if needed.

---

## 🔹 Key Concepts in Implementation

### **a) `dim=-1`**

* Normalization is done **across the last dimension** (i.e., across features of each token embedding).
* For input shape `(batch_size, seq_len, d_model)`, this means each token embedding of size `d_model` is normalized independently.

### **b) `keepdim=True`**

* Keeps the normalized dimension for broadcasting during subtraction/division.
* Without it, mean and std would have shape `(batch_size, seq_len)` → cannot broadcast to `(batch_size, seq_len, d_model)`.

### **c) `nn.Parameter`**

* `alpha` and `bias` are **learnable parameters**.
* PyTorch tracks gradients for them, so they are updated during training.
* This allows LayerNorm to **scale and shift** normalized embeddings, giving flexibility to the model.

---

## 🔹 Example Calculation

Suppose a token embedding:

```
x = [1.0, 2.0, 3.0]
```

1. **Mean**:
   $ \mu = (1 + 2 + 3)/3 = 2 $

2. **Standard deviation**:
   $ \sigma = \sqrt{((1-2)^2 + (2-2)^2 + (3-2)^2)/3} = 0.8165 $

3. **Normalized vector**:
$
   [
   \hat{x} = [(1-2)/0.8165, (2-2)/0.8165, (3-2)/0.8165] = [-1.2247, 0, 1.2247]
   ]
$
4. **After learnable scale and shift** (`alpha=1, beta=0` initially):
$
   [
   y = [-1.2247, 0, 1.2247]
   ]
$
✅ Output now has **zero mean** and **unit variance**.

---

## 🔹 Visual Summary

```
Input Token Embedding:        Normalized + Scaled/Shifted Output:
[1.0, 2.0, 3.0]             [-1.2247, 0, 1.2247]
[2.0, 0.5, 1.5]             [ 1.1355, -1.1136, -0.0219]
...
```

* Each token embedding is **normalized independently across its features**.
* LayerNorm ensures consistent scale for activations across layers, helping the network learn more effectively.

---

## ✅ Summary

* LayerNorm stabilizes training by normalizing **token embeddings across features**.
* `dim=-1` ensures normalization along the feature axis; `keepdim=True` ensures correct broadcasting.
* Learnable parameters `alpha` and `bias` allow the model to adapt normalization as needed.
* Essential in Transformer architectures for **stable, efficient, and consistent training**.



---


# FeedForward Block in Transformers

The **FeedForward Block** is a key component in Transformer architectures. After attention layers, each token embedding is passed through a **position-wise fully connected network** to increase model capacity and allow **non-linear feature transformations**.

* Attention layers capture **relationships between tokens**, but do **not mix features within a token**.
* The feedforward network (FFN) allows the model to **process each token embedding individually** and apply **non-linear transformations**.
* Helps the model **learn complex mappings** beyond linear attention.

---

## 🔹 Architecture and Shape Explanation

### **Forward pass shapes:**

```
Input:  x ∈ ℝ^(Batch, seq_len, d_model)
Step 1: Linear_1 → ℝ^(Batch, seq_len, d_ff)
Step 2: ReLU + Dropout
Step 3: Linear_2 → ℝ^(Batch, seq_len, d_model)
Output: Same shape as input
```

**Why these shapes?**

1. **`d_model → d_ff`**:

   * Expands the embedding to a higher-dimensional space (`d_ff > d_model`) to capture richer, more complex features.
   * Think of it like projecting a 3D point into a higher-dimensional space to make it more linearly separable.

2. **`d_ff → d_model`**:

   * Projects it back to the original embedding size so it can be **added back via residual connection** in the Transformer.
   * Ensures the feedforward block is **position-wise** and does not change the overall shape of the sequence representation.

---

## 🔹 Mathematical Formula

For a token embedding $ x \in \mathbb{R}^{d_\text{model}} $:
$
\text{FFN}(x) = W_2 (\text{Dropout}(\text{ReLU}(W_1 x + b_1))) + b_2
$
Where:

* $ W_1 \in \mathbb{R}^{d_\text{ff} \times d_\text{model}} ,  b_1 \in \mathbb{R}^{d_\text{ff}} $
* $ W_2 \in \mathbb{R}^{d_\text{model} \times d_\text{ff}} ,  b_2 \in \mathbb{R}^{d_\text{model}} $
* ReLU introduces **non-linearity**
* Dropout provides **regularization**

✅ This is applied **independently to each token** in the sequence (hence shape `(Batch, seq_len, ...)`).

---

## 🔹 Example

Suppose we have:

* Batch size = 2
* Sequence length = 3
* `d_model = 4`
* `d_ff = 8`

```
Input x (2, 3, 4):

[
 [[0.1, 0.2, 0.3, 0.4],
  [0.5, 0.6, 0.7, 0.8],
  [0.9, 1.0, 1.1, 1.2]],

 [[1.3, 1.4, 1.5, 1.6],
  [1.7, 1.8, 1.9, 2.0],
  [2.1, 2.2, 2.3, 2.4]]
]
```

1. **Linear_1 (`d_model → d_ff`)** → shape `(2, 3, 8)`
2. **ReLU + Dropout** → elementwise non-linearity and random regularization
3. **Linear_2 (`d_ff → d_model`)** → shape `(2, 3, 4)`

✅ Output has **same shape as input** and is ready for **residual connection + LayerNorm** in the Transformer.

---

## ✅ Summary

* FeedForward blocks are **position-wise MLPs** applied independently to each token.
* `d_model → d_ff → d_model` allows **expansion to learn richer features** and projection back for residual connection.
* ReLU introduces **non-linear transformation**, and Dropout provides **regularization**.
* Maintains the **original sequence shape**, ensuring it can be added back to attention outputs in Transformers.

---
