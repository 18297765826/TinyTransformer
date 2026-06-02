import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self,dropout=0.1):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        dim = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = self.softmax(scores)
        attention = self.dropout(attention)
        out = torch.matmul(attention, value)
        return out,attention


