#手撕transformer第一天
#20260603


import torch
import torch.nn as nn
from MHA import MultiHeadAttention
from Encoder import FFN

class decoder_layer(nn.Module):
    def __init__(self,num_head,dim,d_ff,dropout=0.1):
        super().__init__()
        self.MHA=MultiHeadAttention(num_head=num_head,dim=dim,dropout=dropout)
        self.cross_MHA=MultiHeadAttention(num_head=num_head,dim=dim,dropout=dropout)
        self.ffn=FFN(dim,d_ff,dropout)
        self.norm1=nn.LayerNorm(dim)
        self.norm2=nn.LayerNorm(dim)
        self.norm3=nn.LayerNorm(dim)
    def forward(self,x,memory,src_mask=None,tgt_mask=None):
        resdual1=x
        out=self.MHA(x,x,x,tgt_mask)
        out=self.norm1(out+resdual1)
        resdual2=out
        out=self.cross_MHA(out,memory,memory,src_mask)
        out=self.norm2(resdual2+out)
        resdual3=out
        out=self.ffn(out)
        out=self.norm3(out+resdual3)
        return out


class Decoder(nn.Module):
    def __init__(self,num_layer,num_head,dim,d_ff,dropout=0.1):
        super().__init__()
        self.layer=nn.ModuleList([decoder_layer(num_head,dim,d_ff,dropout) for _ in range(num_layer)])
    def forward(self,x,memory,src_mask,tgt_mask):
        for layer in self.layer:
            x=layer(x,memory,src_mask,tgt_mask)
        return x
