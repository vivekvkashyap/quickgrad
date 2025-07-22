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

    # Add __iadd__ to support += operations
    # def __iadd__(self, other):
    #     if isinstance(other, vector):
    #         self.data += other.data
    #     else:
    #         self.data += other
    #     return self

    def __add__(self, other):
        other = other if isinstance(other, vector) else vector(other)
        
        # Handle broadcasting in forward pass
        out = vector(self.data + other.data, (self, other), '+')
        
        def _backward():
            # Handle broadcasting in backward pass
            self_grad = out.grad
            other_grad = out.grad
            
            # If self was broadcast, sum gradients back to original shape
            if self.data.shape != out.grad.shape:
                # Sum along broadcast dimensions and reshape
                self_grad = np.sum(out.grad, axis=None, keepdims=True)
                while self_grad.ndim > len(self.data.shape):
                    self_grad = np.squeeze(self_grad, axis=0)
                self_grad = np.broadcast_to(self_grad, self.data.shape)
                
                # Alternative: use proper axis reduction
                axes_to_sum = []
                # Find axes that were broadcast
                ndim_diff = len(out.grad.shape) - len(self.data.shape)
                if ndim_diff > 0:
                    axes_to_sum.extend(range(ndim_diff))
                
                # Sum along broadcasted dimensions
                for i in range(len(self.data.shape)):
                    if self.data.shape[i] == 1 and out.grad.shape[i + ndim_diff] > 1:
                        axes_to_sum.append(i + ndim_diff)
                
                if axes_to_sum:
                    self_grad = np.sum(out.grad, axis=tuple(axes_to_sum), keepdims=True)
                    self_grad = self_grad.reshape(self.data.shape)
            
            # If other was broadcast, sum gradients back to original shape  
            if other.data.shape != out.grad.shape:
                other_grad = np.sum(out.grad, axis=None, keepdims=True)
                while other_grad.ndim > len(other.data.shape):
                    other_grad = np.squeeze(other_grad, axis=0)
                other_grad = np.broadcast_to(other_grad, other.data.shape)
                
                # Alternative: use proper axis reduction
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
    

    # def __add__(self, other):
    #     other = other if isinstance(other, vector) else vector(other)
    #     # broadcast_shape = np.broadcast_shapes(self.shape(), other.shape())
    #     # self, other = self.broadcast_to(broadcast_shape), other.broadcast_to(broadcast_shape)
    #     out = vector(self.data + other.data, (self, other), '+')
    
    #     def _backward():
    #         # if self.data.shape != out.grad.shape:
    #             # self.grad = self.grad + np.sum(out.grad * np.ones_like(self.data))
    #             # other.grad = other.grad + np.sum(out.grad * np.ones_like(other.data))
    #         # else:
    #         # self.grad = np.expand_dims(self.grad,0)
    #         # other.grad = np.expand_dims(other.grad, 0)
    #         self.grad +=  out.grad
    #         other.grad +=  out.grad
    #     out._backward = _backward
    #     return out

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
    
    # def broadcast_to(self, target_shape):
    #     input_shape = self.shape()
    #     out = vector(np.broadcast_to(self.data, target_shape), (self,), 'broadcast')

    #     def _backward():
    #         print(input_shape)
    #         print('bcf')
    #         print(target_shape)
    #         print('oo')
    #         broadcast_axes_result = get_broadcast_axes(input_shape, target_shape)
    #         broadcast_axes = broadcast_axes_result[0]
    #         print('abc')
    #         print(broadcast_axes_result)
    #         # if broadcast_axes and isinstance(broadcast_axes[0], list):
    #         #     broadcast_axes = broadcast_axes[0]
    #         # ans = out.grad
    #         # reduced_grad = ans
            
    #         # if broadcast_axes:  
    #         #     axes_tuple = tuple(broadcast_axes)
    #         #     reduced_grad = np.sum(ans, axis=axes_tuple, keepdims=True)
    #         #     reduced_grad = reduced_grad.reshape(input_shape)
    #         # self.grad += reduced_grad
            
    #     out._backward = _backward
    #     return out
    
    def broadcast_to(self, shape):

        data = np.broadcast_to(self.data, shape)
        out = vector(data,  _children=(self, ), _op="broadcast")
        broadcasted_axes = broadcast_axis(self.shape(), shape)[0] # we are interested in left axes

        def _broadcast_backward():
            self.grad += np.sum(out.grad, axis=broadcasted_axes)
            
        out.grad_fn = _broadcast_backward

        return out
    
    def expand_dims(self, axis):
        if isinstance(self, vector):
            return vector(np.expand_dims(self.data, axis=axis))
        else:
            return vector(np.expand_dims(self, axis=axis))
        # self = self if isinstance(self, vector) else vector(self)
        # return vector(np.expand_dims(self, axis=axis))
    
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

    # def __2dmatmul__(self, other):
    #     out = vector(np.matmul(self.data, other.data), (self, other), '@')

    #     def _backward():
    #         other_data = other.transpose((1,0)).data
    #         self_data = self.transpose((1,0)).data
    #         self.grad += np.matmul(out.grad, other_data)
    #         other.grad += np.matmul(self_data, out.grad)

    #     out._backward = _backward
    #     return out

    # def __matmul__(self, other):
        
    #     # assert self._d == other._d, "Tensors must be of the same type i.e. numpy or mlx"
    #     other = other if isinstance(other, vector) else vector(other, use_np=self.is_np_tensor)
        
    #     out = vector(self.data @ other.data, _children=(self, other), _op='@')
    #     if not self.requires_grad and not other.requires_grad:
    #         return out

    #     le_axis = (0, ) if self.data.ndim == 1 else ()
    #     re_axis = (-1, ) if other.data.ndim == 1 else ()
    #     rese_axis = le_axis + re_axis
    #     l, r = broadcast_axis(self.data.shape[:-2], other.data.shape[:-2])

    #     def _matmul_backward():
    #         if self.requires_grad:
    #             self.grad = np.reshape(
    #                 np.sum(
    #                     np.expand_dims(out.grad, axis=rese_axis) @
    #                     np.expand_dims(other.data, axis=re_axis).swapaxes(-1, -2),
    #                     axis = l
    #                 ),
    #                 self.data.shape
    #             )
    #         if other.requires_grad:
    #             other.grad = np.reshape(
    #                 np.sum(
    #                     np.expand_dims(self.data, axis=le_axis).swapaxes(-1, -2) @
    #                     np.expand_dims(out.grad, axis=rese_axis),
    #                     axis = r
    #                 ),
    #                 other.data.shape
    #             )

    #     out.grad_fn = _matmul_backward
    #     return out
    
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
            # Gradient flows through where input is in valid range
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
        
        # FIXED: Keep grad as numpy array, not vector
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

# Helper function (you'll need to implement this based on your needs)
def get_broadcast_axes(input_shape, target_shape):
    """
    Returns the axes that were broadcasted from input_shape to target_shape
    """
    # Simple implementation - you may need to adjust based on your specific needs
    input_ndim = len(input_shape)
    target_ndim = len(target_shape)
    
    # Pad input_shape with 1s at the beginning if needed
    padded_input = (1,) * (target_ndim - input_ndim) + input_shape
    
    broadcast_axes = []
    for i, (inp_dim, tgt_dim) in enumerate(zip(padded_input, target_shape)):
        if inp_dim == 1 and tgt_dim > 1:
            broadcast_axes.append(i)
    
    return [broadcast_axes], []  # Return in expected format

def broadcast_axis(left, right):
    """
    mlx uses broadcasting before performing array ops
    this function determines which axes on either arrays will be broadcasted
    in order to calculate gradients along those axes.

    example:
    >>> left.shape = (3, 1)
    >>> right.shape = (1, 4)
    >>> broadcast_axis(left, right)     # ((1, ), (0, ))

    here the second axis for left, and first axis for right will be broadcasted
    """
    
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
    
    # def __lt__(self, other): 
    #     return self.data < (other.data if isinstance(other, Number) else other)

    # def __le__(self, other): 
    #     return self.data <= (other.data if isinstance(other, Number) else other)
    
    # def __eq__(self, other): 
    #     return self.data == (other.data if isinstance(other, Number) else other)
    
    # def __hash__(self):
    #     return id(self)
    
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
