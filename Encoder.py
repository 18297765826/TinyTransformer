#手撕transformer第一天
#20260603



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import dropout

from MHA import MultiHeadAttention

class FFN(nn.Module):
    def __init__(self,dim,d_ff,dropout=0.1):
        super().__init__()
        self.fc1=nn.Linear(dim,d_ff)
        self.fc2=nn.Linear(d_ff,dim)
        self.dropout=nn.Dropout(dropout)
        self.layernorm=nn.LayerNorm(dim)
        # self.layernorm=nn.LayerNorm(-1)要传最后一个维度的大小
    def forward(self,x):
        # resdual = x
        # x=self.layernorm(self.fc2(F.relu(self.fc1(x))))
        # return self.dropout(resdual+x)
        #post norm
        x=self.fc2(self.dropout(F.relu(self.fc1(x))))
        # x=self.layernorm(x+resdual)
        return x



class EncoderLayer(nn.Module):
    def __init__(self,num_head,dim,d_ff,dropout=0.1):
        super().__init__()
        self.MHA=MultiHeadAttention(num_head,dim,dropout)
        self.ffn=FFN(dim,d_ff,dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self,x,mask=None):
        resdual1 = x
        out=self.MHA(x,x,x,mask)
        out=self.norm1(resdual1+out)
        resdual2=out
        out=self.ffn(out)
        out=self.norm2(out+resdual2)
        return out






