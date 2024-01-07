from .autodiff import Operation
from .utils import *
import dark.tensor as dt

# ------ scalar ------
class AbsoluteValue(Operation):

    def forward(self, x):
        return dt.abs(x)

    def backward(self, dldy, y, x):
        gz = (x > 0)
        lz = dt.logical_not(gz)
        return [dldy * gz - dldy * lz]

class Add(Operation):

    def forward(self, a, b):
        return a + b

    def backward(self, dldy, y, a, b):
        return reduce_sum(dldy, a.shape), reduce_sum(dldy, b.shape)

class Subtract(Operation):

    def forward(self, a, b):
        return a - b

    def backward(self, dldy, y, a, b):
        return reduce_sum(dldy, a.shape), reduce_sum(-dldy, b.shape)

class Mul(Operation):

    def forward(self, a, b):
        return a * b

    def backward(self, dldy, y, a, b):
        dlda = dldy * b
        dldb = dldy * a

        return reduce_sum(dlda, a.shape), reduce_sum(dldb, b.shape)

class Divide(Operation):

    def forward(self, a, b):
        return a / b

    def backward(self, dldy, y, a, b):
        dlda = dldy / b
        dldb = -dldy * a / dt.square(b)

        return reduce_sum(dlda, a.shape), reduce_sum(dldb, b.shape)

class Exp(Operation):

    def forward(self, x):
        return dt.exp(x)

    def backward(self, dydl, y, x):
        return [y * dydl]

class Logarithm(Operation):

    def forward(self, x):
        return dt.log(x)

    def backward(self, dldy, y, x):
        return [dldy / x]
    
class Tanh(Operation):
    
    def forward(self, x):
        return dt.tanh(x)
    
    def backward(self, dldy, y, x):
        return [dldy * (1 - y * y)]

class Max(Operation):

    def forward(self, a, b):
        return dt.maximum(a, b)

    def backward(self, dldy, y, a, b):
        c = a > b
        dlda = dldy * c
        dldb = dldy * dt.logical_not(c)
        
        return dlda, dldb

class Min(Operation):

    def forward(self, a, b):
        return dt.minimum(a, b)

    def backward(self, dldy, y, a, b):
        c = a < b
        dlda = dldy * c
        dldb = dldy * dt.logical_not(c)
        return dlda, dldb

class Pow(Operation):

    def forward(self, x, n):
        return dt.power(x, n)

    def backward(self, dldy, y, x, n):
        return [n * dt.power(x, n - 1) * dldy]

class SquareRoot(Operation):

    def forward(self, x):
        return dt.sqrt(x)

    def backward(self, dldy, y, x):
        return [.5 * dldy / y]


# ------ transformation & logical ------
class View(Operation):

    def forward(self, x, **kwargs):
        outShape = kwargs["shape"]
        return dt.reshape(x, outShape)

    def backward(self, dldy, y, x):
        origShape = x.shape
        return [dt.reshape(dldy, origShape)]

class Transpose(Operation):

    def forward(self, x):
        y = dt.transpose(x)
        return y

    def backward(self, dldy, y, x):
        o = dt.transpose(dldy)
        return [o]

class Cat(Operation):

    def forward(self, *inputs, **kwargs):
        self.dim = kwargs['dim']
        return dt.concatenate(inputs, self.dim)

    def backward(self, dldy, y, *inputs):
        indices = [x.shape[self.dim] for x in inputs]
        indices = dt.cumsum(dt.array(indices))
        indices = [int(x) for x in indices]

        result = dt.split(dldy, indices[:-1], axis=self.dim)
        return result  
        
class Reshape(Operation):

    def forward(self, input, **kwargs):
        self.out_shape = kwargs["shape"]  
        out = input.reshape(self.out_shape)

        return out
    
    def backward(self, grad, out, input):
        dldy = grad.reshape(input.shape)

        return [dldy]
    
class MoveAxis(Operation):

    def forward(self, input, **kwargs):
        self.src_dim = kwargs["source"]
        self.tgt_dim = kwargs["destination"]

        out = dt.moveaxis(input, self.src_dim, self.tgt_dim)
        return out
    
    def backward(self, grad, out, input):
        dldy = dt.moveaxis(grad, self.tgt_dim, self.src_dim)

        return [dldy]
    

# ------ matrix-reduction ------
class Mean(Operation):

    def forward(self, x, **kwargs):
        return dt.mean(x, **kwargs, keepdims=True)

    def backward(self, dldy, y, x):
        norm = dt.prod(dt.array(x.shape)) / dt.prod(dt.array(dldy.shape))
        
        return [dldy * dt.ones(x.shape) / norm]
    
class Var(Operation):

    def forward(self, x, **kwargs):
        self.dim = kwargs["axis"]
        return dt.var(x, self.dim, keepdims=True)

    # https://math.stackexchange.com/questions/2836083/derivative-of-the-variance-wrt-x-i
    def backward(self, dldy, y, x):
        norm = dt.prod(dt.array(x.shape)) / dt.prod(dt.array(dldy.shape))
        
        m = dt.mean(x, self.dim, keepdims=True)
        return [dldy * 2 * (x - m) / norm]

class Sum(Operation):

    def forward(self, x, **kwargs):
        return dt.sum(x, **kwargs, keepdims=True)

    def backward(self, dldy, y, x):
        return [dldy * dt.ones(x.shape)]


# ------ matrix ------
class MatMul(Operation):

    def forward(self, a, b):
        y = dt.matmul(a, b)
        return y

    def backward(self, dldy, y, a, b):
        dlda = dt.matmul(dldy, b.T)
        dldb = dt.matmul(a.T, dldy)
        return dlda, dldb
    
# https://hackmd.io/@machine-learning/blog-post-cnnumpy-fast
# https://stackoverflow.com/questions/34254679/how-can-i-implement-deconvolution-layer-for-a-cnn-in-numpy
# https://coolgpu.github.io/coolgpu_blog/github/pages/2020/10/04/convolution.html
class Conv2d(Operation):

    def forward(self, x, k, **kwargs):
        self.padding = kwargs["padding"]
        self.stride = kwargs["stride"]
        return dt.conv2d(x, k, self.stride, self.padding)

    def backward(self, dldy, y, x, k):        
        dldx, dldk = dt.conv2d_grad(dldy, x, k, self.stride, self.padding)        
        return dldx, dldk

class MaxPool2d(Operation):

    def forward(self, x, **kwargs):
        self.kernel_size = kwargs["kernel_size"]
        self.stride = kwargs["stride"]
        res, self.locs = dt.max_pool2d(x, self.kernel_size, self.stride)
        return res

    def backward(self, dldy, y, x):
        dldx = dt.max_unpool2d(dldy, self.locs, x.shape, self.kernel_size, self.stride)
        return [dldx]
    
class Dropout(Operation):
    
    def forward(self, x, p):
        self.p = p
        self.mask = dt.random.binomial(1, 1 - p, size=x.shape)
        
        result = self.mask * x
        if self.p < 1.0:
            result /= (1 - self.p)
            
        return result
    
    def backward(self, dldy, y, x):
        result = dldy * self.mask
        if self.p < 1.0:
            result /= (1 - self.p)
            
        return result 



def abs(x):
    return AbsoluteValue.apply(x)

def add(a, b):
    return Add.apply(a, b)

def div(a, b):
    return Divide.apply(a, b)

def exp(x):
    return Exp.apply(x)

def log(x):
    return Logarithm.apply(x)

def tanh(x):
    return Tanh.apply(x)

def matmul(a, b):
    return MatMul.apply(a, b)

def transpose(x):
    return Transpose.apply(x)

def max(a, b):
    return Max.apply(a, b)

def mean(x, dim=None):
    return Mean.apply(x, axis=dim)

def var(x, dim=0):
    return Var.apply(x, axis=dim)

def min(a, b):
    return Min.apply(a, b)

def pow(x, n):
    return Pow.apply(x, n)

def subtract(a, b):
    return Subtract.apply(a, b)

def sum(x, dim=None):
    return Sum.apply(x, axis=dim)

def sqrt(x):
    return SquareRoot.apply(x)

def mul(a, b):
    return Mul.apply(a, b)

def neg(x):
    return Subtract.apply(0, x)

def view(x, shape):
    return View.apply(x, shape=shape)

def conv2d(s, k, stride = 1, padding = 0):
    return Conv2d.apply(s, k, padding = padding, stride=stride)

def max_pool2d(x, kernel_size = 2, stride = 2):
    return MaxPool2d.apply(x, kernel_size = kernel_size, stride = stride)

def cat(inputs, dim = 0):
    return Cat.apply(*inputs, dim=dim)

def dropout(x, p=0.2):
    return Dropout.apply(x, p=p)

def reshape(x, shape):
    return Reshape.apply(x, shape=shape)

def moveaxis(x, source, destination):
    return MoveAxis.apply(x, source=source, destination=destination)