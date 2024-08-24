import torch
from torch import nn
import numpy as np

class InputLayer(nn.Module):
    
    def __init__(self, d_model, vocabulary_size):
        super().__init__()
        self.d_model = d_model
        self.vocabulary_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, d_model)

    def forward(self):
        return self.embedding * np.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pr = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = []



print(
    torch.arange(0, 512)
    ) 