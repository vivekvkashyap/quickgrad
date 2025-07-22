from ..core import vector
from ._module import Module
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features):
        k = 1 / in_features
        self.w = vector(np.random.uniform(-k, k, (out_features, in_features)))
        self.b = vector(np.random.uniform(-1, 1, (out_features, )))
    
    def forward(self, x):
        w_x = x @ self.w.transpose((1,0)) 
        out = w_x + self.b.broadcast_to(w_x.shape())
        return out
    
    # def parameters(self):
    #     return [self.w, self.b]