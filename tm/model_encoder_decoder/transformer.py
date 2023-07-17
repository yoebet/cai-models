import copy, math
import torch
from torch import nn
from tm.model.model_config import ModelConfig
from tm.model.model_util import clones, Generator, LayerNorm, SublayerConnection, attention, PositionWiseFeedForward


class Transformer(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pe,
                 tgt_pe,
                 generator
                 ):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pe = src_pe
        self.tgt_pe = tgt_pe
        self.generator = generator

    def forward(self,
                src,  # (batch,s seq_len,d_input)
                tgt,  # (batch,t seq_len,d_input)
                src_mask,  # (batch,1,seq_len)
                tgt_mask  # (batch,seq_len,seq_len)
                ):
        es = self.encode(src, src_mask)  # (batch,seq_len,d_model)
        rr = self.decode(es, src_mask, tgt, tgt_mask)
        return rr  # (batch,seq_len,d_model)

    def encode(self,
               src,  # (batch,s seq_len,d_input)
               src_mask
               ):
        er = self.src_pe(src)
        # er: (batch,seq_len,d_model)
        rr = self.encoder(er, src_mask)
        # rr: (batch,seq_len,d_model)
        return rr

    def decode(self,
               memory,  # (batch,seq_len,d_model)
               src_mask,  # (batch,1,10)
               tgt,  # tensor([[1]])
               tgt_mask  # tensor([[[True]]])
               ):
        er = self.tgt_pe(tgt)  # (batch,t seq_len,d_model)
        rr = self.decoder(er, memory, src_mask, tgt_mask)
        return rr  # (batch,t seq_len,d_model)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self,
                x,  # (batch,seq_len,d_model)
                mask
                ):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self,
                 size,
                 self_attn,
                 feed_forward,
                 dropout
                 ):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self,
                x,  # (batch,seq_len,d_model)
                mask
                ):
        # torch.jit.frontend.UnsupportedNodeError: Lambda aren't supported:
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # x: # (batch,seq_len,d_model)
        rr = self.sublayer[1](x, self.feed_forward)
        # rr # (batch,seq_len,d_model)
        return rr


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self,
                x,  # (batch,tgt seq_len,d_model)
                memory,  # (batch,seq_len,d_model)
                src_mask,
                tgt_mask  # tensor([[[True]]])
                ):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
            # (batch,t seq_len,d_model)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self,
                x,  # (batch,tgt seq_len,d_model)
                memory,  # (batch,seq_len,d_model)
                src_mask,
                tgt_mask  # tensor([[[True]]])
                ):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # x: (batch,q seq_len,d_model)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # x: (batch,q seq_len,d_model)
        r = self.sublayer[2](x, self.feed_forward)
        # (batch,1,d_model)
        return r


class MultiHeadedAttention(nn.Module):
    def __init__(self,
                 h,
                 d_model,  # 512
                 dropout=0.1,
                 device=None
                 ):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # 64
        self.h = h  # 8
        self.linears = clones(nn.Linear(d_model, d_model, device=device), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                query,  # (batch,seq_len,d_model)
                key,  # ..
                value,  # ..
                mask=None
                ):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)  # 1
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        # q,k,v: (batch,heads,seq_len,d_head)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x: (batch,heads,q seq_len,d_head)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)  # (batch,seq_len,d_model)
        rr = self.linears[-1](x)  # (batch,q seq_len,d_model)
        return rr


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model,
                 dropout,
                 max_len,
                 device=None
                 ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(device)  # (1,max_len,d_model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self,
                x  # (batch,seq_len,d_model)
                ):
        pr = self.pe[:, :x.size(1)].clone().detach()
        # pr (batch,seq_len,d_model)
        x = x + pr
        return self.dropout(x)


def make_model(config: ModelConfig,
               device=None
               ):
    c = copy.deepcopy
    d_input = config.d_input
    d_model = config.d_model
    heads = config.heads
    blocks = config.blocks
    max_len = config.max_len
    dropout = config.dropout
    d_ff = config.d_ff
    d_gen_ff = config.d_gen_ff

    attn = MultiHeadedAttention(heads, d_model, device=device)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout, device=device)
    position = PositionalEncoding(d_model, dropout, max_len=max_len, device=device)
    model = Transformer(
        # encoder
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), blocks),
        # decoder
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), blocks),
        # src_pe
        nn.Sequential(nn.Linear(d_input, d_model, dtype=torch.float, device=device), c(position)),
        # tgt_pe
        nn.Sequential(nn.Linear(d_input, d_model, dtype=torch.float, device=device), c(position)),
        # generator
        Generator(d_model, d_gen_ff=d_gen_ff, device=device))
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
