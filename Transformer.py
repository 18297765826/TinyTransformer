"""
踩坑记录

mask 要在 softmax 前填 -inf
多头注意力拆分后要 contiguous() 才能 view
两个子层的 LayerNorm 必须独立
register_buffer 用于位置编码这类固定张量
CrossEntropyLoss 已含 softmax，输出层不要重复加
"""



from Decoder import Decoder
from Encoder import Encoder
import torch.nn as nn
import torch
import math

class Pos_emb(nn.Module):
    def __init__(self, dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, dim)                                        # [max_len, dim]
        position = torch.arange(0, max_len).unsqueeze(1)                      # [max_len, 1]
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)          # [1, max_len, dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, dim]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)




class transformer(nn.Module):
    def __init__(self,vocab_size,num_layer,num_head,dim,d_ff,dropout=0.1):
        super().__init__()
        self.encoder=Encoder(num_layer,num_head,dim,d_ff,dropout=dropout)
        self.decoder=Decoder(num_layer,num_head,dim,d_ff,dropout=dropout)
        self.token_emb=nn.Embedding(vocab_size,dim)
        self.pos_emb=Pos_emb(dim,vocab_size,dropout)
        self.linear=nn.Linear(dim,vocab_size)
    def forward(self,x,tgt,src_mask=None,tgt_mask=None):
        x=self.token_emb(x)
        x=self.pos_emb(x)
        memory =self.encoder(x,src_mask)
        tgt=self.token_emb(tgt)
        tgt=self.pos_emb(tgt)
        out=self.decoder(tgt,memory,src_mask,tgt_mask)
        out=self.linear(out)
        return out




# 超参数
vocab_size = 5000
num_layer  = 6
num_head   = 8
dim        = 512
d_ff       = 2048
dropout    = 0.1
batch_size = 2
src_len    = 128
tgt_len    = 128

# causal mask：防止 decoder 看到未来的 token
def make_causal_mask(size):
    mask = torch.tril(torch.ones(size, size))  # 下三角矩阵
    return mask.unsqueeze(0).unsqueeze(0)      # [1, 1, size, size]

tgt_mask = make_causal_mask(tgt_len)



# 构建模型
model = transformer(vocab_size, num_layer, num_head, dim, d_ff, dropout)
model.eval()

# 随机生成 token id
src = torch.randint(0, vocab_size, (batch_size, src_len))  # [2, 10]
tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))  # [2, 8]

with torch.no_grad():
    out = model(src, tgt, tgt_mask=tgt_mask)

print("输入 src shape:", src.shape)   # [2, 10]
print("输入 tgt shape:", tgt.shape)   # [2, 8]
print("输出 shape:", out.shape)        # [2, 8, 1000]
print("带 mask 输出 shape:", out.shape)  # [2, 8, 1000]
print("测试通过！")


from torchinfo import summary

model = transformer(vocab_size, num_layer, num_head, dim, d_ff, dropout)

summary(model, input_data=(src, tgt))