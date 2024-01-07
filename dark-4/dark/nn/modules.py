import dark.tensor as dt
from dark import Parameter, Node, Constant
import dark
from .module import *


class ZeroParam(Parameter):
    def __init__(self, *shape):
        val = dt.zeros(shape)
        super().__init__(val)

class Flatten(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, input):
        batchCnt = input.data.shape[0] if isinstance(input, Node) else input.shape[0] #if the flatten layer is the first then input is numpy array 
        result = dark.view(input, (batchCnt, -1))
        return result

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = ZeroParam(out_features, in_features)
        self.bias    = ZeroParam(1, out_features)

    def forward(self, x):
        result = dark.add(dark.matmul(x, dark.transpose(self.weights)), self.bias)
        return result

class Sigmoid(Module):
    def __init__(self):
       super().__init__()
    
    def forward(self, x):
        den = dark.add(1, dark.exp(dark.neg(x)))
        return dark.div(1, den)

class ReLU(Module):
    def __init__(self):
       super().__init__()

    def forward(self, x):
        result = dark.max(x, dt.zeros(x.data.shape))
        return result
    
class LeakyReLU(Module):
    def __init__(self, slope = 0.2):
       super().__init__()
       self.slope = slope

    def forward(self, x):
        result = dark.max(x, dark.mul(x, self.slope))
        return result

class Softmax(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        n = dark.exp(x)
        d = dark.sum(n, dim=self.dim)
        result = dark.div(n, d)
        return result