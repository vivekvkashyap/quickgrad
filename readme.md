# Quickgrad

**Quickgrad** is a simple autograd engine built from scratch using NumPy. It supports both numbers and tensors, and helps you understand how automatic differentiation works under the hood, just like in PyTorch, but much simpler and easier to follow.

---

## Repository Structure

``` 
quickgrad/
│
├── core/
│ ├── init.py
│ └── engine.py # defines the Tensor class and its supported ops
│
├── nn/ # Neural network layers
│ ├── linear.py
│ ├── layernorm.py
│ ├── attention.py
│ ├── dropout.py
│ ├── _acts_modules.py # Activation functions
│ └── _losses.py # Loss functions
│
├── optim/ # optimization algo (e.g., SGD, Adam)
│ └── ...
│
└── simple_neural_network.ipynb # example notebook using a simple MLP for classification
``` 

Refer to [`./quickgrad/core/engine.py`](./quickgrad/core/engine.py) for the `vector` class, which implements all core tensor operations and autograd logic.

---

## Quickstart

### Create and Use Tensors

```python
import numpy as np
from quickgrad import vector

a = vector(np.random.randn(2, 2))
b = vector(np.random.randn(2, 5))
c = a @ b
print(c.shape)
```


### Define a Model (similar to PyTorch)

```python
import quickgrad.nn as nn
model = nn.Sequential(
    nn.Linear(in_features, 5),
    nn.ReLU(),
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, out_features),
    nn.Sigmoid()
)
```

### Train the Model

```python
from quickgrad import optim
from quickgrad.nn._losses import BCELoss
import numpy as np

opt = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = BCELoss()
losses = []

for epoch in range(epochs):
    X_train.requires_grad = False  
    y_pred = model.forward(X_train)
    loss = loss_fn.forward(y_train, y_pred)
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    losses.append(loss.data.item())
```



---

## Features

- Full support for scalar and tensor operations
- Autograd via backpropagation
- Neural network layers: Linear, ReLU, Sigmoid, LayerNorm, etc.
- Loss functions like MSE and BCE
- Optimizers including SGD and Adam
- PyTorch-style APIs (`Sequential`, `forward`, `.parameters()`)
- lightweight and built using NumPy

---

## To Do

- [] Add GPU support
- [ ] Add Distributed Data Parallel (DDP) support
- [ ] Improve internal implementations of `__add__` and `__matmul__`
- [ ] Add examples for attention models

---

## Acknowledgements

Inspired by:
- [micrograd](https://github.com/karpathy/micrograd) by **Andrej Karpathy** (scalar only autograd)
- [smolgrad](https://github.com/smolorg/smolgrad) by Maharshi (tensor autograd and repository structure).

---



