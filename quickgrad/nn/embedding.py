from ..core import vector
from ._module import Module
import numpy as np

class Embedding(Module):
    def __init__(self, num_emb, emb_dim):
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.embedding = vector(np.random.normal(size=(num_emb, emb_dim)))

    def forward(self, x):
        out = self.embedding[x.data]
        return out