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


# Multi-Head Attention in Transformers

**Multi-Head Attention (MHA)** allows the Transformer to **attend to different parts of the sequence simultaneously**, capturing diverse relationships between tokens. Each “head” learns a different attention pattern.

* Standard attention computes relationships between **queries, keys, and values** but can only focus in **one representation subspace**.
* Multi-head attention splits the embedding into multiple subspaces (`head`) to capture **different types of dependencies** in parallel.
* Enables richer modeling of **contextual relationships** between tokens.

---

## 🔹 Key Components

### **a) Linear Projections**

For input token embeddings (X \in \mathbb{R}^{seq_len \times d_\text{model}}), we compute:

$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$

Where:

* $ W_Q, W_K, W_V \in \mathbb{R}^{d_\text{model} \times d_\text{model}} $
* Queries (Q), Keys (K), Values (V) all have shape `(Batch, seq_len, d_model)` initially.

---

### **b) Splitting into multiple heads**

* `d_model` is divided by `head` → `d_k = d_model / head`
* Each head has its **own subspace**, shape `(Batch, head, seq_len, d_k)`

This is done via:

```
query = query.view(batch, seq_len, head, d_k).transpose(1, 2)
key   = key.view(batch, seq_len, head, d_k).transpose(1, 2)
value = value.view(batch, seq_len, head, d_k).transpose(1, 2)
```

✅ Now, attention is computed **independently for each head**.

---

## 🔹 Scaled Dot-Product Attention

For each head, attention is computed as:

$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$

* $ Q K^\top $ → computes similarity between tokens
* Scale by $ \frac{1}{\sqrt{d_k}} $ to prevent large dot products
* Softmax → converts scores to probabilities
* Multiply by V → weighted sum of values

If a mask is provided (e.g., for padding or causal attention), the scores for masked positions are set to (-\infty) before softmax.

---

## 🔹 Combining Heads

After attention for each head:

1. Transpose back: `(Batch, head, seq_len, d_k)` → `(Batch, seq_len, head, d_k)`
2. Concatenate all heads → `(Batch, seq_len, d_model)`

This gives a **single combined representation** that encodes multiple attention patterns.

3. Multiply by output weight `W_O`:

$
\text{Output} = \text{ConcatHeads} \cdot W_O
$

* $ W_O \in \mathbb{R}^{d_\text{model} \times d_\text{model}} $
* Ensures output has **same shape as input** `(Batch, seq_len, d_model)` for residual connections.

---

## 🔹 Forward Pass Shapes

| Step                           | Shape                             |
| ------------------------------ | --------------------------------- |
| Input embeddings `x`           | `(Batch, seq_len, d_model)`       |
| Linear projections Q/K/V       | `(Batch, seq_len, d_model)`       |
| Split heads                    | `(Batch, head, seq_len, d_k)`     |
| Attention scores               | `(Batch, head, seq_len, seq_len)` |
| Weighted sum (attended values) | `(Batch, head, seq_len, d_k)`     |
| Concatenate heads              | `(Batch, seq_len, d_model)`       |
| Final Linear `W_O`             | `(Batch, seq_len, d_model)`       |

---

## 🔹 Intuition

* **Each head** can attend to different parts of the sequence: one might focus on previous tokens, another on syntactic dependencies.
* **Splitting and projecting** into multiple heads allows the model to **capture diverse relationships simultaneously**.
* **Output projection (`W_O`)** ensures that combined attention is compatible with the rest of the Transformer (residual connections + layer norm).

---

## ✅ Summary

* Multi-head attention enables **parallel, multi-subspace attention** on token embeddings.
* Scaled dot-product attention computes **weighted sums of value vectors** based on query-key similarity.
* Heads are concatenated and projected back to `d_model` to **preserve input shape**.
* Essential for **context-aware embeddings** in Transformers.

---
]

# Residual Connection + Layer Normalization in Transformers

The `ResidualConnection` block wraps around sublayers (like **Multi-Head Attention** or **Feed Forward**) to stabilize training and preserve information flow.

* Deep networks suffer from **vanishing gradients** and unstable training.
* **Residual connections** (introduced in ResNets) let layers learn corrections instead of full transformations.
* **Layer normalization** keeps values stable by normalizing across features.
* **Dropout** prevents overfitting.

Together, they make Transformer training **stable, faster, and more robust**.

---

## 🔹 Formula

For input $x$ and sublayer function $ \text{sublayer}(\cdot) $:

$
\text{Output} = x + \text{Dropout}(\text{sublayer}(\text{LayerNorm}(x)))
$

* $x$: Input tensor
* `LayerNorm(x)`: Normalizes activations to stabilize learning
* `sublayer`: Can be **MultiHeadAttention** or **FeedForward**
* `Dropout`: Randomly drops activations during training
* Residual: Adds original input (x) back to the sublayer output

---

## 🔹 Implementation Details

### **a) Layer Normalization**

Normalizes input across the **embedding dimension**:

$
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta
$

* $\mu$: mean of features
* $\sigma$: standard deviation of features
* $\gamma, \beta$: learnable parameters

✅ Ensures stable scale across tokens.

---

### **b) Residual Connection**

Instead of forcing the sublayer to learn **entire mapping**, we just learn:

$
F(x) + x
$

This allows **gradient flow** directly through the skip path.

---

### **c) Dropout**

Applied to the sublayer’s output before adding to (x).
Reduces **overfitting** by preventing reliance on specific neurons.

---

## 🔹 Forward Pass Shapes

| Step                  | Shape                       |
| --------------------- | --------------------------- |
| Input $x$             | `(Batch, seq_len, d_model)` |
| LayerNorm($x$)          | `(Batch, seq_len, d_model)` |
| Sublayer( $LN(x) $)       | `(Batch, seq_len, d_model)` |
| Dropout(...)          | `(Batch, seq_len, d_model)` |
| Residual add with $x$ | `(Batch, seq_len, d_model)` |

✅ The shape is preserved → makes residual connections possible.

---

## 🔹 Intuition

* Residuals allow **deeper networks** without gradient vanishing.
* LayerNorm ensures **numerical stability** across long sequences.
* Dropout adds **regularization**.
* Together, this makes training Transformers **efficient and stable**.

---

## ✅ Summary

* **Residual**: keeps original signal + learned corrections.
* **LayerNorm**: normalizes across features per token.
* **Dropout**: improves generalization.
* Used around **each sublayer** in the Transformer (Attention & FeedForward).

---

# Transformer Encoder Block & Encoder

The **EncoderBlock** and **Encoder** form the foundation of the Transformer. They combine **Self-Attention**, **Feed Forward networks**, and **Residual Connections** to build contextualized representations of sequences.

---

## 🔹 Encoder Block

### **Structure**

Each `EncoderBlock` contains:

1. **Multi-Head Self-Attention** (with masking)
2. **Feed Forward Network**
3. **Residual Connections + Layer Normalization**

---

### **Formula**

For input $x$ and mask $M$:

$
x' = x + \text{Dropout}(\text{MHA}(\text{LayerNorm}(x), M))
$

$
x'' = x' + \text{Dropout}(\text{FFN}(\text{LayerNorm}(x')))
$

Where:

* MHA = Multi-Head Self-Attention
* FFN = Feed Forward Network
* Residual + LayerNorm = stability + gradient flow
* $M$ = attention mask (explained below 👇)

---

### **Why Do We Need Masking?**

In **attention**, we compute:

$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$

* Without masking → every token attends to every other token.
* With masking → we control **which tokens are visible**.

#### ✅ Types of Masks

1. **Padding Mask**

   * Used in the **Encoder**.
   * Ensures the model ignores `<PAD>` tokens (added to equalize sequence lengths).
   * Prevents attention from wasting capacity on meaningless padding.

   Example:

   ```
   Input: ["I", "love", "AI", "<PAD>", "<PAD>"]
   Mask:  [1,   1,     1,      0,      0 ]
   ```

   Attention scores for `<PAD>` will be set to `-∞` before softmax, so probability → 0.

2. **Causal Mask (Look-Ahead Mask)**

   * Used in the **Decoder**.
   * Ensures each position only attends to **previous tokens** (important in autoregressive generation).
   * Prevents “cheating” by looking at future tokens.

---

### **Intuition**

* **Padding Mask**: “Don’t pay attention to fake/padded tokens.”
* **Causal Mask**: “Only attend to the past, not the future.”
* Together, they make attention **valid, efficient, and meaningful**.

---

## 🔹 Encoder

The **Encoder** stacks multiple `EncoderBlock`s and applies a final **LayerNorm**.

### **Formula**

For $L$ layers:

$
x^{(0)} = \text{Input}
$

$
x^{(l)} = \text{EncoderBlock}^{(l)}(x^{(l-1)}, M), \quad l = 1 \dots L
$

$
\text{Output} = \text{LayerNorm}(x^{(L)})
$

Where $M$ = padding mask.

---

### **Shapes**

| Step                    | Shape                       |
| ----------------------- | --------------------------- |
| Input `x`               | `(Batch, seq_len, d_model)` |
| After Block 1           | `(Batch, seq_len, d_model)` |
| After Block 2           | `(Batch, seq_len, d_model)` |
| ... After L blocks      | `(Batch, seq_len, d_model)` |
| Final Normalized Output | `(Batch, seq_len, d_model)` |

---

## 🔹 Intuition

* **Self-Attention** builds contextualized embeddings (each token attends to others).
* **Masking** ensures irrelevant tokens (like padding) are ignored.
* **Residual + LayerNorm** make training stable.
* Stacking blocks lets the encoder build **hierarchical sequence representations**.

---

## ✅ Summary

* **EncoderBlock** = Self-Attention + FeedForward + residuals.
* **Encoder** = Stack of EncoderBlocks + final normalization.
* **Masking** ensures attention ignores padding and maintains valid dependencies.

---

# Transformer Decoder Block & Decoder

The **Decoder** is the second half of the Transformer. It takes **encoder outputs** and **target tokens** to generate predictions step by step.

---

## 🔹 Decoder Block

Each `DecoderBlock` has **3 main parts**:

1. **Masked Self-Attention** (over target sequence → ensures autoregressive generation).
2. **Cross-Attention** (attends to encoder outputs → lets the decoder use source information).
3. **Feed Forward Network** (applies non-linear transformation).

Each part is wrapped with **Residual + LayerNorm**.

---

### **Formula**

For input $x$ (decoder input embeddings) and encoder output $E$:

1. **Masked Self-Attention**
$
   x' = x + \text{Dropout}(\text{MHA}*{\text{self}}(\text{LayerNorm}(x), M*{\text{tgt}}))
$

Here $M_{\text{tgt}}$ = **causal mask** → prevents attending to future tokens.

2. **Cross-Attention**
   $
   x'' = x' + \text{Dropout}(\text{MHA}*{\text{cross}}(\text{LayerNorm}(x'), E, E, M*{\text{src}}))
   $

Here $M_{\text{src}}$ = **padding mask** for encoder outputs.

3. **Feed Forward Network**
   $
   x''' = x'' + \text{Dropout}(\text{FFN}(\text{LayerNorm}(x'')))
   $

---

### **Why Two Types of Attention?**

* **Self-Attention (Decoder side)**: Looks at already generated target tokens, but **causally masked** so token $t$ only attends to $[0, …, t]$.
* **Cross-Attention**: Lets the decoder attend to encoder outputs → links source and target.
* Together: Decoder can **remember past target words** while **referencing the input sequence**.

---

### **Why Masking in Decoder?**

1. **Causal Mask (Target Mask)**

   * Ensures autoregressive generation.
   * Example: When predicting the 4th word, only first 3 words are visible.
   * Prevents "cheating" by looking at future words.

   ```
   Target: [I, love, AI, models]
   At step 3 ("AI"), model can only see [I, love].
   ```

2. **Source Mask**

   * Same as in Encoder → prevents attending to `<PAD>` tokens in encoder outputs.

---

## 🔹 Decoder

The `Decoder` stacks multiple `DecoderBlock`s and applies a final **LayerNorm**.

### **Formula**

$
x^{(0)} = \text{Input (shifted target embeddings)}
$

$
x^{(l)} = \text{DecoderBlock}^{(l)}(x^{(l-1)}, E, M_{\text{src}}, M_{\text{tgt}})
$

$
\text{Output} = \text{LayerNorm}(x^{(L)})
$

Where:

* $E$ = encoder outputs
* $M_{\text{src}}$ = source padding mask
* $M_{\text{tgt}}$ = causal + padding mask for target

---

## 🔹 Shapes

| Step                                   | Shape                           |
| -------------------------------------- | ------------------------------- |
| Input (Decoder Embeddings)             | `(Batch, tgt_seq_len, d_model)` |
| After Masked Self-Attn                 | `(Batch, tgt_seq_len, d_model)` |
| After Cross-Attn (with encoder output) | `(Batch, tgt_seq_len, d_model)` |
| After FFN                              | `(Batch, tgt_seq_len, d_model)` |
| After L blocks                         | `(Batch, tgt_seq_len, d_model)` |
| Final Output (normalized)              | `(Batch, tgt_seq_len, d_model)` |

---

## 🔹 Intuition

* **Masked Self-Attention**: Decoder builds up its own context one token at a time.
* **Cross-Attention**: Decoder aligns target tokens with encoder’s source representation.
* **Residual + LayerNorm**: Stabilize training.
* **Stacking Layers**: Builds hierarchical understanding for text generation.

---

## ✅ Summary

* **DecoderBlock** = Masked Self-Attn + Cross-Attn + FFN.
* **Decoder** = Stack of DecoderBlocks + final normalization.
* **Masking is critical** →

  * Target mask (causal) → ensures autoregressive left-to-right generation.
  * Source mask → ignores padding in encoder outputs.

---

Perfect 👍 let’s break this down like we did for embeddings — step by step explanation, **without PyTorch implementation** (since you already have it), and I’ll also explain **what `dim = -1` means**.

---

#  Projection Layer (a.k.a. Output Layer in Transformers)

## 🔹 Why Projection is Needed

* The transformer processes tokens into hidden vectors of size `d_model` (e.g., 512 or 768).
* But the final output must be a **probability distribution over the vocabulary** (say vocab size = 30,000).
* So we need a projection from **d_model → vocab_size**.
* This is done by a **linear layer (matrix multiplication + bias)**.

Mathematically:
$
\text{logits} = X \cdot W + b
$

Where:

* $X$ = hidden states → shape: **(batch, seq_len, d_model)**
* $W$ = projection weights → shape: **(d_model, vocab_size)**
* $b$ = bias → shape: **(vocab_size, )**
* Output = **(batch, seq_len, vocab_size)**

---

## 🔹 Why Softmax (and log_softmax) is Applied

* After projection, we have **raw scores (logits)** for each token in the vocabulary.
* To interpret them as probabilities, we apply **softmax**:

$
P(y_t = k \mid X) = \frac{e^{\text{logit}_k}}{\sum_j e^{\text{logit}_j}}
$

* In practice, PyTorch often uses **`log_softmax`** because:

  * It is numerically more stable.
  * Training with **negative log-likelihood loss (NLLLoss)** expects log probabilities.

So the output is:
$
\text{log_probs} = \text{log_softmax}(\text{proj}(x), ; \text{dim=-1})
$

---

## 🔹 What does `dim = -1` mean?

* `dim=-1` tells PyTorch **along which axis to apply softmax**.
* `-1` means **the last dimension of the tensor** (Python indexing rule).

Since the output shape is:
**(batch, seq_len, vocab_size)**

* `dim=-1` → softmax is applied **across the vocab dimension**.
* This ensures that for each token position, the sum of probabilities over all vocabulary words = **1**.

Example:

If output = `(2, 5, 10000)`

* batch = 2
* seq_len = 5
* vocab_size = 10000

Then softmax is applied across those **10000 values per token position**, making them valid probabilities.

---

## 🔹 Without dim=-1 (wrong case)

* If you applied softmax on `dim=1` (sequence dimension), you’d get probabilities across sequence length (nonsense).
* If you applied on `dim=0` (batch), you’d normalize across different samples (also nonsense).

Hence, `dim=-1` is **critical**:
It ensures we normalize **only across the vocabulary dimension**, which is what we need in language modeling.

---

✅ **Summary**:

* Projection layer maps hidden vectors → vocabulary logits.
* `log_softmax` converts logits → log-probabilities (stable training).
* `dim=-1` ensures normalization is applied across the **vocab dimension**, not batch or sequence.

---


#  **Transformer Model**

This `Transformer` class combines **Encoder + Decoder + Input Embeddings + Positional Encodings + Projection Layer** into the complete seq2seq model.

---

## 1. **Inputs & Components**

* **Encoder**: Encodes the source sequence into contextual representations.
* **Decoder**: Generates the target sequence step by step, attending to both its own past outputs and the encoder outputs.
* **Embeddings**:

  * `src_embed`: Converts source tokens → dense vectors.
  * `tgt_embed`: Converts target tokens → dense vectors.
* **Positional Encodings**:

  * `src_pos`: Adds order information to source embeddings.
  * `tgt_pos`: Adds order info to target embeddings.
* **Projection Layer**: Maps decoder outputs → vocabulary probabilities.

So the Transformer = **[Embeddings + Positional Encoding] → Encoder → Decoder → Projection → Probabilities**.

---

## 2. **encode()**

```python
def encode(self, src, src_mask):
    src = self.src_embed(src)          # Token → Vector
    src = self.src_pos(src, src_mask)  # Add positional encoding
    return self.encoder(src, src_mask) # Encoder outputs
```

* Takes a source sentence.
* Converts it into embeddings + adds position info.
* Runs through **encoder stack**.
* Returns **encoder outputs**, which summarize the entire source sequence.

---

## 3. **decode()**

```python
def decode(self, encoder_output, src_mask, tgt, tgt_mask):
    tgt = self.tgt_embed(tgt)          # Token → Vector
    tgt = self.tgt_pos(tgt)            # Add positional encoding
    return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
```

* Takes:

  * `encoder_output` (from `encode()`).
  * Partial target sequence (so far generated).
  * `tgt_mask` (causal mask → ensures decoder can’t peek at future tokens).
* Runs through **decoder stack**:

  * First attends to itself (**masked self-attention**).
  * Then attends to encoder outputs (**cross-attention**).
* Produces decoder hidden states.

---

## 4. **project()**

```python
def project(self, x):
    return self.projection_layer(x)
```

* Maps decoder hidden states → logits → log-probabilities over target vocabulary.
* This is the final step before sampling/greedy decoding/beam search.

---

## 5. **BuildTransformer Function**

This function actually **constructs the Transformer architecture**.

### (a) Embedding Layers

* `InputEmbeddings(d_model, vocab_size)`
* Learnable embeddings for both **src** and **tgt** vocabularies.

### (b) Positional Encoding

* `PositionalEncoding(d_model, seq_len, dropout)`
* Since transformers have no recurrence, positional encoding injects **order info**.

### (c) Encoder Blocks

* Repeats `N` times:

  1. **Multi-Head Self-Attention** → lets tokens attend to each other.
  2. **Feed Forward Block** → adds depth & non-linearity.
  3. **Residual connections + LayerNorm** inside `EncoderBlock`.

### (d) Decoder Blocks

* Repeats `N` times:

  1. **Masked Multi-Head Self-Attention** → attends only to past tokens.
  2. **Cross-Attention** → attends to encoder outputs.
  3. **Feed Forward Block**.
  4. **Residual + LayerNorm**.

### (e) Encoder & Decoder

* Wrap blocks in `Encoder` and `Decoder`.

### (f) Projection Layer

* `ProjectionLayer(d_model, tgt_vocab_size)` converts hidden states to vocab logits.

### (g) Xavier Initialization

* Applies `nn.init.xavier_uniform_` to all weight matrices.
* Ensures stable variance propagation during training.

---

## 6. **Overall Workflow**

1. Input sequence (src) → **Encoder** produces context.
2. Partial target sequence (tgt so far) → **Decoder** (with masking + cross-attention).
3. Decoder output → **Projection Layer** → vocab probabilities.
4. Next token predicted → fed back into decoder → repeat.

---

✅ **Summary:**

* The Transformer is a **seq2seq architecture**: encoder encodes, decoder decodes with attention.
* `BuildTransformer` constructs it by stacking encoder/decoder blocks, embeddings, and projection.
* `encode()` handles source input, `decode()` handles target prediction, `project()` maps to vocab.
* Key ideas: **self-attention, cross-attention, masking, positional encoding, projection**.

---
