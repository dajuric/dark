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


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding = 0, stride = 1):
        super().__init__()

        self.weights = ZeroParam(out_channels, in_channels, kernel_size, kernel_size)
        self.bias    = ZeroParam(1,            out_channels, 1,          1)

        self.padding = padding
        self.stride = stride

    def forward(self, input):
        return dark.add(dark.conv2d(input, self.weights, self.stride, self.padding), self.bias)

class MaxPool2d(Module):
    def __init__(self, kernel_size = 2, stride = -1):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride if stride > 0 else kernel_size

    def forward(self, input):
        return dark.max_pool2d(input, self.kernel_size, self.stride)
    
# https://stackoverflow.com/questions/64364320/how-to-implement-batchnorm2d-in-pytorch-myself
class BatchNorm2d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        assert isinstance(dim, int)
        dim = (1, dim, 1, 1)

        self.eps = eps           * dt.ones((1, 1, 1, 1))
        self.momentum = momentum * dt.ones((1, 1, 1, 1))

        # parameters (trained with backprop)
        self.gamma = Parameter(dt.ones(dim))
        self.beta = Parameter(dt.zeros(dim))

        # buffers (trained with a running 'momentum update')
        self.running_mean = dt.zeros(dim)
        self.running_var = dt.ones(dim)

    def forward(self, x):
        is_training = any([x for x in [self.gamma, self.beta] if x.requires_grad])
        
        if is_training:
            xmean = dark.mean(x, (0, 2, 3)) # batch mean
            xvar = dark.var(x, (0, 2, 3)) # batch variance 
        else:
            xmean = self.running_mean
            xvar = self.running_var
            
        xhat = dark.div(dark.subtract(x, xmean), dark.sqrt(dark.add(xvar, self.eps))) # normalize to unit variance
        out = dark.add(dark.mul(self.gamma, xhat), self.beta)

        # update the buffers
        if is_training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.data
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * xvar.data

        return out
    
class Dropout(Module):
    def __init__(self, p = 0.2):
        super().__init__()

        self.p = p
      
    def forward(self, input):
        return dark.dropout(input, self.p)