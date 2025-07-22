from typing import *

from ..core import vector
from ._module import Module
from ._activations import *


class ReLU(Module):
    def __init__(self):
        pass

    def forward(self, x):
        out = relu(x)
        return out
    
class Sigmoid(Module):
    def __init__(self) -> None:
        pass

    def forward(self, x):
        out = sigmoid(x)
        return out
    
class Softmax(Module):
    def __init__(self, dim: int = None) -> None:
        self.dim = dim or -1

    def forward(self, x):
        out = softmax(x, axis = self.dim)
        return out
    
class GeLU(Module):
    def __init__(self, approx):
        super().__init__()
        self.approx = approx
    
    def forward(self, x):
        if self.approx == 'tanh':
            return self.tanh_approx(x)
        else:
            return self.gelu(x)
    
    def gelu(self, x):
        y =(x * 0.7978845608 * (1 + 0.044715 * x * x))
        return 0.5 * x * (1 + tanh(y))
    
    def tanh_approx(self, x):
        y = (0.7978845608 * (x + 0.044715 * x * x * x))
        return 0.5 * x * (1 + tanh(y))