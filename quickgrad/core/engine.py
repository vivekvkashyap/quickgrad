import math
import numpy as np
from typing import *
import random

class vector:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data, dtype=np.float64)  # Keep as numpy array
        self.label = label
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
        self.requires_grad = True

    def _reset_grad(self):
        self.grad = np.zeros_like(self.data)

    def __repr__(self):
        return f"vector(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, vector) else vector(other)
        
        out = vector(self.data + other.data, (self, other), '+')
        
        def _backward():
        
            self_grad = out.grad
            other_grad = out.grad
            
            if self.data.shape != out.grad.shape:
            
                self_grad = np.sum(out.grad, axis=None, keepdims=True)
                while self_grad.ndim > len(self.data.shape):
                    self_grad = np.squeeze(self_grad, axis=0)
                self_grad = np.broadcast_to(self_grad, self.data.shape)
            
                axes_to_sum = []
                ndim_diff = len(out.grad.shape) - len(self.data.shape)
                if ndim_diff > 0:
                    axes_to_sum.extend(range(ndim_diff))
                
                for i in range(len(self.data.shape)):
                    if self.data.shape[i] == 1 and out.grad.shape[i + ndim_diff] > 1:
                        axes_to_sum.append(i + ndim_diff)
                
                if axes_to_sum:
                    self_grad = np.sum(out.grad, axis=tuple(axes_to_sum), keepdims=True)
                    self_grad = self_grad.reshape(self.data.shape)
             
            if other.data.shape != out.grad.shape:
                other_grad = np.sum(out.grad, axis=None, keepdims=True)
                while other_grad.ndim > len(other.data.shape):
                    other_grad = np.squeeze(other_grad, axis=0)
                other_grad = np.broadcast_to(other_grad, other.data.shape)
                
                axes_to_sum = []
                ndim_diff = len(out.grad.shape) - len(other.data.shape)
                if ndim_diff > 0:
                    axes_to_sum.extend(range(ndim_diff))
                
                for i in range(len(other.data.shape)):
                    if other.data.shape[i] == 1 and out.grad.shape[i + ndim_diff] > 1:
                        axes_to_sum.append(i + ndim_diff)
                
                if axes_to_sum:
                    other_grad = np.sum(out.grad, axis=tuple(axes_to_sum), keepdims=True)
                    other_grad = other_grad.reshape(other.data.shape)
            
            self.grad += self_grad
            other.grad += other_grad
            
        out._backward = _backward
        return out
    
    def exp(self):
        out = vector(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = vector(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            other_data = other
        elif hasattr(other, 'data'):
            other_data = other.data
        else:
            other_data = other
            
        out = vector(np.power(self.data, other_data), (self,), '_pow')

        def _backward():
            self.grad += (other_data * (self.data ** (other_data - 1))) * out.grad
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=True):
        out = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = vector(out, (self,), 'sum')

        def _backward():
            self.grad += np.ones_like(self.data, dtype=np.float64) * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, vector) else vector(other)
        other = other.broadcast_to(self.shape())
        out = vector(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad
            
        out._backward = _backward
        return out

    
    def broadcast_to(self, shape):

        data = np.broadcast_to(self.data, shape)
        out = vector(data,  _children=(self, ), _op="broadcast")
        broadcasted_axes = broadcast_axis(self.shape(), shape)[0] 
        def _broadcast_backward():
            self.grad += np.sum(out.grad, axis=broadcasted_axes)
            
        out.grad_fn = _broadcast_backward

        return out
    
    def expand_dims(self, axis):
        if isinstance(self, vector):
            return vector(np.expand_dims(self.data, axis=axis))
        else:
            return vector(np.expand_dims(self, axis=axis))
    
    def transpose(self, axis):
        out = vector(np.transpose(self.data, axes=axis), (self,), 'T')

        def _backward():
            self.grad += np.transpose(out.grad, axes=axis)
        out._backward = _backward
        return out

    def var(self, axis):
        mean = self.mean(axis)    
        new_ans = mean.broadcast_to(self.shape())
        ans = (self - mean) ** 2
        numer = ans.sum(axis=axis, keepdims=False)
        denom = self.shape()[axis] 
        out = (numer / denom)
        return out 

    def std(self, axis):
        return self.var(axis) ** (1/2)
        
    def mean(self, axis=None):
        n = self.shape()[axis] if axis is not None else self.data.size
        out = self.sum(axis, keepdims=True) * (n ** -1)
        return out

    def reshape(self, target_shape):
        out = vector(np.reshape(self.data, target_shape))

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def __setitem__(self, index, val):
        self.data[index] = val.data.copy()
        self.grad[index] = val.grad.copy()
    
    def __getitem__(self, index):
        out = vector(self.data[index], (self,), 'get_item')

        def _backward():
            self.grad[index] += out.grad
        out._backward = _backward
        return out

    # def split(self, indices, axis):
    #     out = vector(np.split(self, indices_or_sections=indices, axis=axis))

    #     def _backward():
    #         grad_pieces = [piece.grad for piece in out]
    #         self.grad += np.concatenate(grad_pieces, axis=axis)

    #     out._backward = _backward
    #     return out

    def concat(self, other, axis):
        self_shape = self.shape()
        axis_self_shape = self_shape[axis]
        out = vector(np.concatenate((self.data, other.data), axis=axis), (self, other), 'cat')

        def _backward():
            self_grad, other_grad = np.array_split(out.grad, [axis_self_shape], axis=axis)
            self.grad += self_grad
            other.grad += other_grad
        out._backward = _backward
        return out

    
    def __matmul__(self, other):
        out = vector(np.matmul(self.data, other.data), (self, other), '@')
        
        def _backward():        
            other_transposed = np.swapaxes(other.data, -2, -1)
            self_transposed = np.swapaxes(self.data, -2, -1)
            if self.requires_grad:
                self.grad += np.matmul(out.grad, other_transposed)
            if other.requires_grad:
                other.grad += np.matmul(self_transposed, out.grad)
        
        out._backward = _backward
        return out
    
    def masked_fill(self, condition, val):
        out = vector(np.where(condition, self.data, val.data))
        
        def _backward():
            self.grad += np.where(condition, 0, out.grad)
            val.grad += np.where(condition, out.grad, 0)
        out._backward = _backward
        return out
    
    def clip(self, min_val, max_val):
        out = vector(min(max(self.data, min_val), max_val))

        def _backward():
            self.grad = np.clip(self.grad, min_val, max_val)
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return vector(other) + (-self)
    
    def __radd__(self, other):
        return vector(other) + self
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return vector(other) * (self ** -1)

    def __rmul__(self, other):
        return self * other

    def shape(self):
        return self.data.shape

    def clamp(self, min_val, max_val):
        clamped_data = np.clip(self.data, min_val, max_val)
        out = vector(clamped_data, (self,), 'clamp')
        
        def _backward():
            mask = (self.data >= min_val) & (self.data <= max_val)
            self.grad += mask * out.grad
        
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

def get_broadcast_axes(input_shape, target_shape):

    input_ndim = len(input_shape)
    target_ndim = len(target_shape)
    
    padded_input = (1,) * (target_ndim - input_ndim) + input_shape
    
    broadcast_axes = []
    for i, (inp_dim, tgt_dim) in enumerate(zip(padded_input, target_shape)):
        if inp_dim == 1 and tgt_dim > 1:
            broadcast_axes.append(i)
    
    return [broadcast_axes], []  

def broadcast_axis(left, right):
    
    ldim = len(left)
    rdim = len(right)
    maxdim = max(ldim, rdim)

    lshape_new = (1, ) * (maxdim - ldim) + left
    rshape_new = (1, ) * (maxdim - rdim) + right

    assert len(lshape_new) == len(rshape_new)

    left_axes, right_axes = [], []

    for i in range(len(lshape_new)):
        if lshape_new[i] > rshape_new[i]:
            right_axes.append(i)
        elif rshape_new[i] > lshape_new[i]:
            left_axes.append(i)

    return tuple(left_axes), tuple(right_axes)

class Number:
    def __init__(self, data, _children = (), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._op = _op
        self.label = label
        self._backward = lambda : None
        self._prev = set(_children)
    
    # def __hash__(self):
    #     return id(self)
    
    def __repr__(self):
        return f"Number(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Number) else Number(other)
        out = Number(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Number) else Number(other)
        out = Number(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out 

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Number(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Number(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t **2) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Number(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad 
        out._backward = _backward
        return out
    
    def log(self):
        x = self.data
        out = Number(math.log(x), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Number(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        x = self.data
        out = Number(1/(1+math.exp(-x)), (self,), 'sigmoid')

        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        return out
    
    def clamp(self, min_val, max_val):
        x = self.data
        clamped = min(max(x, min_val), max_val)
        out = Number(clamped, (self,), 'clamp')

        def _backward():
            if min_val < x < max_val:
                self.grad += out.grad
            else:
                self.grad += 0.0
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __radd__(self, other): 
        return self + other

    def __rtruediv__(self, other):  
        return (self ** -1) * other
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
