#手撕transformer第一天
#20260603

import torch
import torch.nn as nn
import math


class Attention(nn.Module):
    def __init__(self,dropout=0.1):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.softmax=nn.Softmax(-1)

        #mask最好给默认值None
    def forward(self,q,k,v,mask=None):
        dim=q.size(-1)
        score=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(dim)
        """
        Softmax 后得到的是概率
        Mask 的目的是让某些位置概率变成 0
        如果先 Softmax，再 Mask，那么剩余概率和不再等于 1
        """
        if mask is not None:
            score=score.masked_fill(mask==0,float("-inf"))
        score=self.softmax(score)
        #错误
        # if mask is not None:
        #     pass
        score=self.dropout(score)
        out=torch.matmul(score,v)
        return out,score


class MultiHeadAttention(nn.Module):
    def __init__(self,num_head,dim,dropout=0.1):
        super().__init__()
        #assert dim/num_head==0
        assert dim % num_head==0
        self.num_head=num_head
        self.dim=dim
        self.head_dim = dim // num_head
        self.dropout=nn.Dropout(dropout)
        self.wq=nn.Linear(dim,dim)
        self.wk=nn.Linear(dim,dim)
        self.wv=nn.Linear(dim,dim)
        self.wo=nn.Linear(dim,dim)
        self.attention=Attention(dropout)
    def forward(self,q,k,v,mask=None):
        batch=q.size(0)
        q,k,v=self.wq(q),self.wk(k),self.wv(v)
        q = q.view(batch,-1,self.num_head,self.head_dim).transpose(1,2)
        k = k.view(batch, -1, self.num_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_head, self.head_dim).transpose(1, 2)
        out,_=self.attention(q,k,v,mask)
        out  = out.transpose(1,2).contiguous().view(batch,-1,self.num_head*self.head_dim)
        out=self.wo(out)
        out=self.dropout(out)
        return out






