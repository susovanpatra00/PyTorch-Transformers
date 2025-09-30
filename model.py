import math
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p= dropout)
    
        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply the sin to even position and cos to odd positiom
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    



class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias



class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features=d_model, out_features=d_ff) # W1 and b1
        self.dropout = dropout
        self.linear_2 = nn.Linear(in_features=d_ff, out_features= d_model)

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model) 
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

        

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, head: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.dropout = dropout
        assert d_model % head == 0, "d_model is not divisible by no of head"

        self.d_k = d_model // head
        self.w_q = nn.Linear(d_model, d_model) # wq
        self.w_k = nn.Linear(d_model, d_model) # wk
        self.w_v = nn.Linear(d_model, d_model) # wv

        self.w_o = nn.Linear(d_model, d_model)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, head, seq_len, d_k) --> # (Batch, head, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, head, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores 



    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model) 
        key = self.w_k(k)    # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model) 
        value = self.w_v(v)  # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model) 

        # (Batch, seq_len, d_model) --> (Batch, seq_len, head, d_k) -->  (Batch, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.head, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.head, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.head, self.d_k).transpose(1, 2)
    
        x, self.attention-scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)


