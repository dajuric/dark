from .autodiff import Operation
from .utils import *
import dark.tensor as dt

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

class MatMul(Operation):

    def forward(self, a, b):
        y = dt.matmul(a, b)
        return y

    def backward(self, dldy, y, a, b):
        dlda = dt.matmul(dldy, b.T)
        dldb = dt.matmul(a.T, dldy)
        return dlda, dldb
    
class Transpose(Operation):

    def forward(self, x):
        y = dt.transpose(x)
        return y

    def backward(self, dldy, y, x):
        o = dt.transpose(dldy)
        return [o]

class Max(Operation):

    def forward(self, a, b):
        return dt.maximum(a, b)

    def backward(self, dldy, y, a, b):
        c = a > b
        dlda = dldy * c
        dldb = dldy * dt.logical_not(c)
        
        return dlda, dldb

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

class Min(Operation):

    def forward(self, a, b):
        return dt.minimum(a, b)

    def backward(self, dldy, y, a, b):
        c = a < b
        dlda = dldy * c
        dldb = dldy * dt.logical_not(c)
        return dlda, dldb

class Mul(Operation):

    def forward(self, a, b):
        return a * b

    def backward(self, dldy, y, a, b):
        dlda = dldy * b
        dldb = dldy * a

        return reduce_sum(dlda, a.shape), reduce_sum(dldb, b.shape)

class Pow(Operation):

    def forward(self, x, n):
        return dt.power(x, n)

    def backward(self, dldy, y, x, n):
        return [n * dt.power(x, n - 1) * dldy]

class Subtract(Operation):

    def forward(self, a, b):
        return a - b

    def backward(self, dldy, y, a, b):
        return reduce_sum(dldy, a.shape), reduce_sum(-dldy, b.shape)

class Sum(Operation):

    def forward(self, x, **kwargs):
        return dt.sum(x, **kwargs, keepdims=True)

    def backward(self, dldy, y, x):
        return [dldy * dt.ones(x.shape)]

class SquareRoot(Operation):

    def forward(self, x):
        return dt.sqrt(x)

    def backward(self, dldy, y, x):
        return [.5 * dldy / y]

class View(Operation):

    def forward(self, x, **kwargs):
        outShape = kwargs["shape"]
        return dt.reshape(x, outShape)

    def backward(self, dldy, y, x):
        origShape = x.shape
        return [dt.reshape(dldy, origShape)]


# https://hackmd.io/@machine-learning/blog-post-cnnumpy-fast
# https://stackoverflow.com/questions/34254679/how-can-i-implement-deconvolution-layer-for-a-cnn-in-numpy
# https://coolgpu.github.io/coolgpu_blog/github/pages/2020/10/04/convolution.html
class Conv2d(Operation):

    def forward(self, x, k, **kwargs):
        self.padding = kwargs["padding"]
        self.stride = kwargs["stride"]
        return dt.conv2d_forward(x, k, self.stride, self.padding)

    def backward(self, dldy, y, x, k):        
        dldx, dldk = dt.conv2d_backward(dldy, x, k, self.stride, self.padding)        
        return dldx, dldk

class MaxPool2d(Operation):

    def forward(self, x, **kwargs):
        self.kernel_size = kwargs["kernel_size"]
        self.stride = kwargs["stride"]
        res, self.locs = dt.maxpool2d_forward(x, self.kernel_size, self.stride)
        return res

    def backward(self, dldy, y, x):
        dldx = dt.maxpool2d_backward(dldy, x, self.locs, self.kernel_size, self.stride)
        return [dldx]
    
# https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11
# https://towardsdatascience.com/only-numpy-understanding-back-propagation-for-transpose-convolution-in-multi-layer-cnn-with-c0a07d191981
class ConvTranspose2d(Operation):

    def forward(self, x, k, **kwargs):  
        self.padding = kwargs["padding"]
        self.stride = kwargs["stride"]
        self.output_padding = kwargs["output_padding"]
           
        xe = self.dilate(x, self.stride)  

        _, _, kh, kw = k.shape   
        result = dt.conv2d_forward(xe, k.transpose(1, 0, 2, 3), 1, kh - 1)  
        result = self.pad(result, xe.shape, k.shape, self.padding, self.output_padding)
        return result

    def backward(self, dldy, y, x, k):
        #pad gradient and convolve w.r.t k
        dldx = dt.conv2d_forward(dldy, k, self.stride, self.padding)
                
        #dilate x, convolve w.r.t dilated x and crop valid part
        xe = self.dilate(x, self.stride)
        dldk = dt.conv2d_forward(dldy.transpose((1, 0, 2, 3)), xe.transpose((1, 0, 2, 3)), 1, self.padding) 
        
        dldk = self.pad(dldk, dldy.shape, xe.shape, 0, 0)
        dldk = dt.swapaxes(dldk, 0, 1)
                
        return dldx, dldk
        pass
        
    def pad(self, o, x_shape, k_shape, padding, output_padding):
        _, _, x_h, x_w = x_shape
        _, _, o_h, o_w = o.shape
        _, _, k_h, k_w = k_shape 

        h_valid = x_h - 2 * padding + (k_h - 1) + output_padding
        w_valid = x_w - 2 * padding + (k_w - 1) + output_padding
        o = o[..., padding:h_valid+padding, padding:w_valid+padding]

        _, _, o_h, o_w = o.shape
        o = dt.pad(o, [(0, 0), (0, 0), (0, o_h - o_h), (0, o_w - o_w)])
        return o
        
    def dilate(self, x, stride):
        tb, tc, th, tw = x.shape
        xe = dt.zeros((tb, tc, (th - 1) * stride + 1, (tw - 1) * stride + 1))
        xe[:, :, ::self.stride, ::self.stride] = x

        return xe
    
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

class Slice(Operation):

    def forward(self, input, **kwargs):
        self.dims = kwargs["dim"]
        out = input[self.dims]

        return out
    
    def backward(self, grad, out, input):
        dldy = dt.zeros(input.shape)
        dldy[self.dims] = grad

        return [dldy]
    
class Mask(Operation):

    def forward(self, input, **kwargs):
        self.mask = kwargs["mask"]
        assert self.mask.shape == input.shape
        
        out = input[self.mask]
        return out
    
    def backward(self, grad, out, input):
        dldy = dt.zeros(input.shape)
        dldy[self.mask] = grad

        return [dldy]
        
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

def conv_transpose2d(s, k, stride = 1, padding = 0, output_padding = 0):
    return ConvTranspose2d.apply(s, k, padding = padding, stride = stride, output_padding = output_padding)

def cat(inputs, dim = 0):
    return Cat.apply(*inputs, dim=dim)

def dropout(x, p=0.2):
    return Dropout.apply(x, p=p)

def mask(x, mask):
    return Mask.apply(x, mask=mask)