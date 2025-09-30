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

* ( pos ) = position index (0,1,...,seq_len-1)
* ( i ) = embedding dimension index (0,...,d_model-1)
* ( d_\text{model} ) = embedding size

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
