import copy
import torch
from torch import nn
from tm.model_decoder_only.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb
from tm.model.model_config import ModelConfig
from tm.model.model_util import clones, Generator, Classifier, LayerNorm, SublayerConnection, attention, \
    PositionWiseFeedForward


class Transformer(nn.Module):
    def __init__(self,
                 decoder,
                 src_ff,
                 generator
                 ):
        super(Transformer, self).__init__()
        self.decoder = decoder
        self.src_ff = src_ff
        self.generator = generator

    def forward(self,
                src,  # (batch,seq_len,d_input)
                src_mask,  # (batch,seq_len,seq_len)
                ):
        rr = self.decode(src, src_mask)
        return rr  # (batch,seq_len,d_model)

    def decode(self,
               src,  # (batch,seq_len,d_model)
               src_mask,  # (batch,1,10)
               ):
        se = self.src_ff(src)  # (batch,t seq_len,d_model)
        rr = self.decoder(se, src_mask)
        return rr  # (batch,t seq_len,d_model)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self,
                x,  # (batch,tgt seq_len,d_model)
                src_mask,
                ):
        for layer in self.layers:
            x = layer(x, src_mask)
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
                src_mask,
                ):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, src_mask))
        # x: (batch,q seq_len,d_model)
        r = self.sublayer[2](x, self.feed_forward)
        # (batch,1,d_model)
        return r


class MultiHeadedAttention(nn.Module):
    def __init__(self,
                 h,
                 d_model,  # 512
                 dropout=0.1,
                 device=None,
                 max_len=128,
                 ):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # 64
        self.h = h  # 8
        self.linears = clones(nn.Linear(d_model, d_model, device=device), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.position_emb = RotaryEmbedding(self.d_k, max_len=max_len, device=device)

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

        kv_seq_len = key.shape[-2]
        cos, sin = self.position_emb(value, seq_len=kv_seq_len)
        position_ids = torch.arange(0, kv_seq_len)
        position_ids = position_ids.unsqueeze(0).view(-1, kv_seq_len)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        # q,k,v: (batch,heads,seq_len,d_head)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x: (batch,heads,q seq_len,d_head)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)  # (batch,seq_len,d_model)
        rr = self.linears[-1](x)  # (batch,q seq_len,d_model)
        return rr


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

    if config.classify:
        generator = Classifier(d_model, d_gen_ff=d_gen_ff, device=device, n=config.classify_n)
    else:
        generator = Generator(d_model, d_gen_ff=d_gen_ff, device=device)

    attn = MultiHeadedAttention(heads, d_model, max_len=max_len, device=device)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout, device=device)
    model = Transformer(
        # decoder
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), blocks),
        # src_ff
        nn.Sequential(nn.Linear(d_input, d_model, device=device)),
        generator
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
