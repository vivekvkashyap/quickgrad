
from typing import *

from ._optimizer import Optimizer
from ..core import vector
import numpy as np


class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.first_momentum = [np.zeros_like(p.data) for p in self.parameters]
        self.second_momentum = [np.zeros_like(p.data) for p in self.parameters]
        
        self.t = 0
    
    def step(self):
        self.t += 1 
        
        for i in range(len(self.parameters)):
            p = self.parameters[i]
            g_t = p.grad
            # print(g_t)
            
            self.first_momentum[i] = self.beta1 * self.first_momentum[i] + (1 - self.beta1) * g_t
            self.second_momentum[i] = self.beta2 * self.second_momentum[i] + (1 - self.beta2) * g_t * g_t
            
            first_momentum_bias_corrected = self.first_momentum[i] / (1 - self.beta1**self.t)
            second_momentum_bias_corrected = self.second_momentum[i] / (1 - self.beta2**self.t)
            update_term = - self.lr * first_momentum_bias_corrected / (np.sqrt(second_momentum_bias_corrected) + self.eps)
            # print(update_term)
            p.data = p.data + update_term