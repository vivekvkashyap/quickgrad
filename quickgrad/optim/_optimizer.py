from typing import *
from ..core import vector

class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self) -> None:
        """
        Update the parameters.
        """
        raise NotImplementedError("The 'step' method must be implemented in child class.")

    def zero_grad(self):
        for p in self.parameters:
            p._reset_grad()