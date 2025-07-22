from ._module import Module
from typing import *


class Sequential(Module):

    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def append(self, module):
        self.modules.append(module)

    def parameters(self):
        p = []
        for m in self.modules:
            p += m.parameters()
        return p
    
    def forward(self, x):
        if len(self.modules) ==0 : return x
        
        for i in range(len(self.modules)):
            mod = self.modules[i]
            # print(mod)
            # print('abc')
            x = mod.forward(x)

        return x
