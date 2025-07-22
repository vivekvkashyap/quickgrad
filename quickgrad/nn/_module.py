from typing import *
from ..core import vector


class Module:
    def __init__(self):
        self.is_training = True

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def get_vectors(self):
        vectors = []

        for _, value in self.__dict__.items():
            if isinstance(value, vector):
                vectors.append(value)
            elif isinstance(value, Module):
                vectors += value.get_vectors()
        return vectors

    def parameters(self):
        return [t for t in self.get_vectors()]
    
    def train(self):
        self.is_training = True
    
    def eval(self):
        self.is_training = False
