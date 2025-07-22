import numpy as np
import math
import random
from typing import *
from ..core import vector


def relu(input):
    out = vector(np.maximum(input.data, np.zeros_like(input.data)), (input,), "relu")
    def relu_backward():
        input.grad += (input.data>0) * out.grad
    out._backward = relu_backward
    return out


def tanh(input):
    exp_input = (2 * input).exp().data
    out = vector((exp_input - 1) / (exp_input + 1), (input,), 'tanh')
    def tanh_backward():
        input.grad += (1 - (out.data**2)) * out.grad
    out._backward = tanh_backward
    return out

def sigmoid(input):
    exp_input = (input * -1).exp().data
    out = vector(1 / (1 + exp_input),(input,),'sigmoid')
    def sigmoid_backward():
        print('bcd')
        input.grad += out.data * (1 - out.data) * out.grad
    out._backward = sigmoid_backward
    return out

def softmax(input, axis=-1):
    # denominator = input.exp().sum(axis=axis).data 
    # out = vector(input.exp().data / denominator)
    max_input = vector(np.max(input.data, axis=axis, keepdims=True))
    numerator = vector((input - max_input).exp().data)
    denominator = numerator.sum(axis=axis, keepdims=True).data
    out = vector(numerator.data / denominator.data)
    return out
