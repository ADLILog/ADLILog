from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable

torch.random.seed = 0
np.random.seed(0)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        Parameters
        ----------
        encoder : nn.Module
            A neural network object that acts as an encoder.
        decoder : nn.Module
            A neural network object that acts as an encoder.
        src_embed : nn.Module
            Source embedding network
        tgt_embed : nn.Module
            Target embedding network
        generator: nn.Module
            Generator network
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.c = 0

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        out = self.encode(src, src_mask)
#         print("The size of x given to the input of the generator is {}".format(out.shape))
        dim1 = out.shape[0]
        out = out.view(dim1, -1)
        return out

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, size, vocab):
        super(Generator, self).__init__()
        self.size = size
        
        self.proj1 = nn.Linear(size, size)
        self.proj2 = nn.Linear(size, size)
        self.proj =  nn.Linear(self.size, 2)

    def forward(self, x):
#         old
#         sliced_x = x[:, 0, :]
#         out = self.proj(sliced_x)
        out = F.relu(self.proj1(x))
        out = F.tanh(self.proj2(out))
        out = self.proj(out)
        return out


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, n_layers):
        """
        Parameters
        ----------
        layer: nn.Module
            Neural network object
        n_layers: int
            Number of layers for the provided layer object
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """Construct a layer norm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        """"
        Parameters
        ----------
        size: int
            Size of the layer
        self_attn : callable
            Attention mechanism function
        feed_forward : callable
            Feed forward function
        dropout: float
            Weight dropout percentage. [0-1]
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        return x


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, n_layers):
        """
        Parameters
        ----------
        layer: nn.Module
            Neural network object
        n_layers: int
            Number of layers for the provided layer object
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """"
        Parameters
        ----------
        size: int
            Size of the layer
        self_attn : callable
            Attention mechanism function
        src_attn : callable
            Attention mechanism function for the source
        feed_forward : callable
            Feed forward function
        dropout: float
            Weight dropout percentage. [0-1]
        """

        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, in_features, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert in_features % h == 0
        # We assume d_v always equals d_k
        self.d_k = in_features // h
        self.h = h
        self.linears = clones(nn.Linear(in_features, in_features), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from in_features => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        x = self.linears[-1](x)

        return x


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, in_features, out_features, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(in_features, out_features)
        self.w_2 = nn.Linear(out_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, in_features, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, in_features)
        self.in_features = in_features

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.in_features)


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, in_features, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, in_features)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_features, 2) *
                             -(math.log(10000.0) / in_features))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
