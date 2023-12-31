import copy
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Generator(nn.Module):
    is_classifier = False

    def __init__(self,
                 d_model,
                 d_gen_ff=64,
                 dropout=0.1,
                 device=None
                 ):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, d_gen_ff, device=device)
        self.proj2 = nn.Linear(d_gen_ff, 1, device=device)
        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_gen_ff, device=device)

    def forward(self,
                x  # (batch,seq_len,d_model)
                ):
        fx = x[:, -1, :]  # (batch,d_model)
        fx = self.proj(fx)
        # fx = self.dropout(fx)
        fx = F.sigmoid(fx)
        fx = self.layer_norm(fx)
        fx = self.proj2(fx)
        # (batch,1)
        fx = fx.squeeze(-1)
        # (batch,)
        return fx


class Classifier(nn.Module):
    is_classifier = True

    def __init__(self,
                 d_model,
                 d_gen_ff=64,
                 dropout=0.1,
                 device=None,
                 n=3,
                 ):
        super(Classifier, self).__init__()
        self.proj = nn.Linear(d_model, n, bias=False, device=device)
        # self.proj2 = nn.Linear(d_gen_ff, n, bias=False, device=device)
        # self.dropout = nn.Dropout(dropout)
        # self.leaky_relu = nn.LeakyReLU(0.02)
        # self.layer_norm = nn.LayerNorm(d_gen_ff, device=device)
        self.n = n

    def forward(self,
                x  # (batch,seq_len,d_model)
                ):
        # fx = x[:, -1, :]  # (batch,d_model)
        fx = x.mean(1)
        fx = self.proj(fx)
        # fx = self.dropout(fx)
        # fx = F.sigmoid(fx)
        # fx = self.leaky_relu(fx)
        # fx = self.layer_norm(fx)
        # fx = self.proj2(fx)
        # fx = F.sigmoid(fx)
        # (batch,n)
        # fx = F.softmax(fx, dim=-1)
        # fx = F.log_softmax(fx, dim=-1)
        return fx


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self,
                x  # (batch,seq_len,d_model)
                ):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        self.to(x.device)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x,  # (batch,seq_len,d_model)
                sublayer
                ):
        nr = self.norm(x)  # (batch,seq_len,d_model)
        sr = sublayer(nr)  # ..
        dr = self.dropout(sr)  # ..
        return x + dr


def subsequent_mask(size):  # size: 1
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # mask: (1,1,1) [[[0]]]
    r = torch.from_numpy(mask) == 0  # tensor([[[True]]])
    return r


def attention(query,  # (batch,heads,seq_len,d_head)
              key,  # ..
              value,  # ..
              mask=None,  # (batch,1,1,q seq_len)
              dropout=None
              ):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
    # scores: (batch,heads,q seq_len,k seq_len)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)  # (batch,heads,q seq_len,k seq_len)
    if dropout is not None:
        p_attn = dropout(p_attn)
    mm = torch.matmul(p_attn, value)  # (batch,heads,q seq_len,d_head)
    return mm, p_attn


class PositionWiseFeedForward(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff,
                 dropout=0.1,
                 device=None
                 ):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, device=device)
        self.w_2 = nn.Linear(d_ff, d_model, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # (batch,seq_len,d_model)
        wr = self.w_1(x)
        tr = F.relu(wr)  # (batch,seq_len,d_ff)
        dr = self.dropout(tr)
        rr = self.w_2(dr)  # (batch,seq_len,d_model)
        return rr
