from .autodiff import Operation
from .utils import *
import dark.tensor as xp

class AbsoluteValue(Operation):

    @staticmethod
    def _f(x):
        return xp.abs(x)

    @staticmethod
    def _df(dldy, y, x):
        gz = (x > 0)
        lz = xp.logical_not(gz)
        return [dldy * gz - dldy * lz]

class Add(Operation):

    @staticmethod
    def _f(a, b):
        return a + b

    @staticmethod
    def _df(dldy, y, a, b):
        return reduce_sum(dldy, a.shape), reduce_sum(dldy, b.shape)

class Divide(Operation):

    @staticmethod
    def _f(a, b):
        return a / b

    @staticmethod
    def _df(dldy, y, a, b):
        dlda = dldy / b
        dldb = -dldy * a / xp.square(b)

        return reduce_sum(dlda, a.shape), reduce_sum(dldb, b.shape)

class Exp(Operation):

    @staticmethod
    def _f(x):
        return xp.exp(x)

    @staticmethod
    def _df(dydl, y, x):
        return [y * dydl]

class Logarithm(Operation):

    @staticmethod
    def _f(x):
        return xp.log(x)

    @staticmethod
    def _df(dldy, y, x):
        return [dldy / x]

class MatMul(Operation):

    @staticmethod
    def _f(a, b):
        y = xp.matmul(a, b)
        return y

    @staticmethod
    def _df(dldy, y, a, b):
        dlda = xp.matmul(dldy, b.T)
        dldb = xp.matmul(a.T, dldy)
        return dlda, dldb

class Max(Operation):

    @staticmethod
    def _f(a, b):
        return xp.maximum(a, b)

    @staticmethod
    def _df(dldy, y, a, b):
        c = a > b
        dlda = dldy * c
        dldb = dldy * xp.logical_not(c)
        return dlda, dldb

class Mean(Operation):

    @staticmethod
    def _f(x, **kwargs):
        return xp.mean(x, **kwargs, keepdims=True)

    @staticmethod
    def _df(dldy, y, x):
        return [dldy * xp.ones(x.shape) / x.size]

class Min(Operation):

    @staticmethod
    def _f(a, b):
        return xp.minimum(a, b)

    @staticmethod
    def _df(dldy, y, a, b):
        c = a < b
        dlda = dldy * c
        dldb = dldy * xp.logical_not(c)
        return dlda, dldb

class Mul(Operation):

    @staticmethod
    def _f(a, b):
        return a * b

    @staticmethod
    def _df(dldy, y, a, b):
        dlda = dldy * b
        dldb = dldy * a

        return reduce_sum(dlda, a.shape), reduce_sum(dldb, b.shape)

class Pow(Operation):

    @staticmethod
    def _f(x, n):
        return xp.power(x, n)

    @staticmethod
    def _df(dldy, y, x, n):
        return [n * xp.power(x, n - 1) * dldy]

class Subtract(Operation):

    @staticmethod
    def _f(a, b):
        return a - b

    @staticmethod
    def _df(dldy, y, a, b):
        return reduce_sum(dldy, a.shape), reduce_sum(-dldy, b.shape)

class Sum(Operation):

    @staticmethod
    def _f(x, **kwargs):
        return xp.sum(x, **kwargs, keepdims=True)

    @staticmethod
    def _df(dldy, y, x):
        return [dldy * xp.ones(x.shape)]

class SquareRoot(Operation):

    @staticmethod
    def _f(x):
        return xp.sqrt(x)

    @staticmethod
    def _df(dldy, y, x):
        return [.5 * dldy / y]

class View(Operation):

    @staticmethod
    def _f(x, **kwargs):
        outShape = kwargs["shape"]
        return xp.reshape(x, outShape)

    @staticmethod
    def _df(dldy, y, x):
        origShape = x.shape
        return [xp.reshape(dldy, origShape)]

class Conv2D(Operation):

    @staticmethod
    def _f(s, k, **kwargs):
        return xp.corr2d(s, k, kwargs["padding"])

    @staticmethod
    def _df(dldy, y, s, k):
        p = Conv2D._get_padding(s.shape[-1], dldy.shape[-1], k.shape[-1])
        dlds = xp.conv2d(dldy, k.transpose((1, 0, 2, 3)), xp.abs(p).item())

        p = Conv2D._get_padding(k.shape[-1], dldy.shape[-1], s.shape[-1])
        dldk = xp.conv2d(dldy.transpose((1, 0, 2, 3)), s.transpose((1, 0, 2, 3)), xp.abs(p).item())
        return dlds, dldk

    @staticmethod
    def _get_padding(o, s, k):
        p = (o - s + k - 1) // 2
        return p
            
class MaxPool2D(Operation):

    @staticmethod
    def _f(x, **kwargs):
        return xp.max_pool_2d(x, kwargs["kernel_size"])

    @staticmethod
    def _df(dldy, y, x):
        n = x.shape[-1] // y.shape[-1]
        dldx = xp.max_unpool_2d(dldy, x, n)
        return [dldx]
    
    
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

def matmul(a, b):
    return MatMul.apply(a, b)

def max(a, b):
    return Max.apply(a, b)

def mean(x, dim=0):
    return Mean.apply(x, axis=dim)

def min(a, b):
    return Min.apply(a, b)

def pow(x, n):
    return Pow.apply(x, n)

def subtract(a, b):
    return Subtract.apply(a, b)

def sum(x, dim=0):
    return Sum.apply(x, axis=dim)

def sqrt(x):
    return SquareRoot.apply(x)

def mul(a, b):
    return Mul.apply(a, b)

def neg(x):
    return Subtract.apply(0, x)

def view(x, shape):
    return View.apply(x, shape=shape)

def conv2d(s, k, padding = 0):
    return Conv2D.apply(s, k, padding = padding)

def max_pool2d(x, kernel_size = 2):
    return MaxPool2D.apply(x, kernel_size = kernel_size)