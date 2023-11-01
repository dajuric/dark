import dark.tensor as dt
from dark import Parameter, Node
import dark
from dark import *

class Module():
    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def forward(self, *x):
        pass

    def __call__(self, *x):
        x = [(Constant(p) if not isinstance(p, Node) else p) for p in x]
        return self.forward(*x)

    def modules(self):
        return self._modules.values()

    def parameters(self):
        params = []

        def get_params(module):
           for p in module._parameters.values():
               params.append(p)

           for m in module._modules.values():    
               get_params(m)

        get_params(self)
        return params

    def train(self):
        for p in self.parameters():
            p.requires_grad = True
        
    def eval(self):
       for p in self.parameters():
            p.requires_grad = False

    def apply(self, apply_func):
        for m in self._all_modules():
            apply_func(m)

    def _all_modules(self):
        modules = []

        def get_modules(module):
            modules.append(module)
            for m in module._modules.values():
                get_modules(m)

        get_modules(self)
        return modules

    def __setattr__(self, key, value) -> None: #TODO: introduce getattr ??
        if isinstance(value, Parameter):
            self._parameters[key] = value
        elif isinstance(value, Module):
            self._modules[key] = value

        super().__setattr__(key, value)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            self._modules['Seq-' + str(i)] = m

    def forward(self, input):
        result = input
        for m in self.modules():
            result = m(result)

        return result

class ModuleList(Module):
    def __init__(self):
        super().__init__()
        self.list = []

    def __getitem__(self, idx):
        return self.list[idx]

    def __len__(self):
        return len(self.list)

    def __iter__(self):
        return iter(self.list)

    def append(self, module):
        key = 'Module-' + str(len(self.list))
        self._modules[key] = module
        
        self.list.append(module)
        return self


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
        assert isinstance(kernel_size, int)

        self.weights = ZeroParam(out_channels, in_channels, kernel_size, kernel_size)
        self.bias    = ZeroParam(1,            out_channels, 1,          1)

        self.padding = padding
        self.stride = stride

    def forward(self, input):
        return dark.add(dark.conv2d(input, self.weights, self.padding, self.stride), self.bias)

class MaxPool2d(Module):
    def __init__(self, kernel_size = 2):
        super().__init__()
        assert isinstance(kernel_size, int)

        self.kernel_size = kernel_size

    def forward(self, input):
        return dark.max_pool2d(input, self.kernel_size)
    
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
    
class ConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, output_padding = 0):
        super().__init__()
        assert isinstance(kernel_size, int)

        self.weights = ZeroParam(in_channels, out_channels, kernel_size, kernel_size)
        self.bias    = ZeroParam(1,           out_channels, 1,          1)

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
      
    def forward(self, input):
        return dark.add(dark.conv_transpose2d(input, self.weights, self.padding, self.stride, self.output_padding), self.bias)
    
class Dropout(Module):
    def __init__(self, p = 0.2):
        super().__init__()

        self.p = p
      
    def forward(self, input):
        return dark.dropout(input, self.p)