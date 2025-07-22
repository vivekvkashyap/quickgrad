from ._module import Module
from .linear import *
from .dropout import *
from ._activations import *

from typing import *
import math

class CausalSelfAttention(Module):
    def __init__(self, context_size, n_heads, d_model, att_dropout = 0.7, lin_dropout = 0.6):
        self.context_size = context_size
        self.n_heads = n_heads
        self.d_model = d_model
        self.att_dropout = att_dropout
        self.lin_dropout = lin_dropout

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads

        self.q = Linear(self.d_model, self.d_model)
        self.k = Linear(self.d_model, self.d_model)
        self.v = Linear(self.d_model, self.d_model)
        self.o = Linear(self.d_model, self.d_model)

        self.attention_drop = Dropout(p=self.att_dropout)
        self.linear_drop = Dropout(p=self.lin_dropout)
        self.mask = vector(np.tril(np.ones((self.context_size, self.context_size))))

    def forward(self, x):
        batch_size, seq_length, d_model  = x.shape()
        q = self.q.forward(x)
        k = self.k.forward(x)
        v = self.v.forward(x)
        q = q.reshape((batch_size, seq_length, self.n_heads, self.head_dim)).transpose((0, 2, 1, 3))
        k = k.reshape((batch_size, seq_length, self.n_heads, self.head_dim)).transpose((0, 2, 1, 3))
        v = v.reshape((batch_size, seq_length, self.n_heads, self.head_dim)).transpose((0, 2, 1, 3))

        kT = k.transpose((0, 1, 3, 2))

        qk = ((q @ kT) * (1 / math.sqrt(self.head_dim)))
        qk_masked = qk.masked_fill(self.mask, vector(float("-inf")))
        softmax_qk = softmax(qk_masked, axis = -1)
        softmax_qk = self.attention_drop.forward(softmax_qk)
        attn = softmax_qk @ v
        out = attn.transpose((0, 2, 1, 3)).reshape((batch_size, seq_length, d_model))
        out = self.o.forward(out)
        return self.linear_drop.forward(out)
