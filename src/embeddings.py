import torch
import torch.nn as nn
import numpy as np

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * np.sqrt(self.d_model)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super(PositionalEmbedding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)

        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10_000) / d_model))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.pe = pe.unsqueeze(0)

        self.register_buffer('positional_embedding', pe)



    def forward(self, x):
        x += self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNormalization, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias
