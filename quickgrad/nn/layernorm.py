
from ..core import vector
from ._module import Module
import numpy as np

class Layernorm(Module):
    def __init__(self, normalized_shape, eps=1e-05):
        batch, seq_length, dim = normalized_shape
        self.eps = eps
        self.scale = vector(np.ones((dim,)))
        self.shift = vector(np.zeros((dim,)))

    def forward(self, x):
        mean = x.mean(axis=-1)
        var = x.var(axis=-1).expand_dims(-1)
        numerator = (x - mean) * self.scale
        denominator = (var + self.eps) ** 1/2
        out = numerator * (denominator ** -1) 
        out = out + self.shift
        return out