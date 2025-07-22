from ..core import vector
from ._module import Module
import numpy as np


class Dropout(Module):
    def __init__(self, p):
        self.p = p

    def forward(self, x):
        # if not self.training:
        #     return x
        mask = np.random.binomial(1, 1-self.p, size=x.shape())
        mask = vector(mask)  
        out = (x * mask) / (1 - self.p)
        return out
