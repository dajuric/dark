from .autodiff import Operation
from .utils import *
import dark.tensor as dt

class AbsoluteValue(Operation):

    @staticmethod
    def _f(x):
        return dt.abs(x)

    @staticmethod
    def _df(dldy, y, x):
        gz = (x > 0)
        lz = dt.logical_not(gz)
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
        dldb = -dldy * a / dt.square(b)

        return reduce_sum(dlda, a.shape), reduce_sum(dldb, b.shape)

class Exp(Operation):

    @staticmethod
    def _f(x):
        return dt.exp(x)

    @staticmethod
    def _df(dydl, y, x):
        return [y * dydl]

class Logarithm(Operation):

    @staticmethod
    def _f(x):
        return dt.log(x)

    @staticmethod
    def _df(dldy, y, x):
        return [dldy / x]

class MatMul(Operation):

    @staticmethod
    def _f(a, b):
        y = dt.matmul(a, b)
        return y

    @staticmethod
    def _df(dldy, y, a, b):
        dlda = dt.matmul(dldy, b.T)
        dldb = dt.matmul(a.T, dldy)
        return dlda, dldb

class Max(Operation):

    @staticmethod
    def _f(a, b):
        return dt.maximum(a, b)

    @staticmethod
    def _df(dldy, y, a, b):
        c = a > b
        dlda = dldy * c
        dldb = dldy * dt.logical_not(c)
        return dlda, dldb

class Mean(Operation):

    @staticmethod
    def _f(x, **kwargs):
        return dt.mean(x, **kwargs, keepdims=True)

    @staticmethod
    def _df(dldy, y, x):
        return [dldy * dt.ones(x.shape) / x.size]
    
class Var(Operation):

    @staticmethod
    def _f(x, **kwargs):
        return dt.var(x, **kwargs, keepdims=True)

    # https://math.stackexchange.com/questions/2836083/derivative-of-the-variance-wrt-x-i
    @staticmethod
    def _df(dldy, y, x, **kwargs):
        dim = kwargs["axis"]
        m = dt.mean(x, dim)
        return [dldy * 2 * (x - m) / x.size]

class Min(Operation):

    @staticmethod
    def _f(a, b):
        return dt.minimum(a, b)

    @staticmethod
    def _df(dldy, y, a, b):
        c = a < b
        dlda = dldy * c
        dldb = dldy * dt.logical_not(c)
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
        return dt.power(x, n)

    @staticmethod
    def _df(dldy, y, x, n):
        return [n * dt.power(x, n - 1) * dldy]

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
        return dt.sum(x, **kwargs, keepdims=True)

    @staticmethod
    def _df(dldy, y, x):
        return [dldy * dt.ones(x.shape)]

class SquareRoot(Operation):

    @staticmethod
    def _f(x):
        return dt.sqrt(x)

    @staticmethod
    def _df(dldy, y, x):
        return [.5 * dldy / y]

class View(Operation):

    @staticmethod
    def _f(x, **kwargs):
        outShape = kwargs["shape"]
        return dt.reshape(x, outShape)

    @staticmethod
    def _df(dldy, y, x):
        origShape = x.shape
        return [dt.reshape(dldy, origShape)]

class Conv2D(Operation):

    @staticmethod
    def _f(s, k, **kwargs):
        return dt.corr2d(s, k, kwargs["padding"])

    @staticmethod
    def _df(dldy, y, s, k):
        p = Conv2D._get_padding(s.shape[-1], dldy.shape[-1], k.shape[-1])
        dlds = dt.conv2d(dldy, k.transpose((1, 0, 2, 3)), dt.abs(p).item())

        p = Conv2D._get_padding(k.shape[-1], dldy.shape[-1], s.shape[-1])
        dldk = dt.conv2d(dldy.transpose((1, 0, 2, 3)), s.transpose((1, 0, 2, 3)), dt.abs(p).item())
        return dlds, dldk

    @staticmethod
    def _get_padding(o, s, k):
        p = (o - s + k - 1) // 2
        return p
            
class MaxPool2D(Operation):

    @staticmethod
    def _f(x, **kwargs):
        return dt.max_pool_2d(x, kwargs["kernel_size"])

    @staticmethod
    def _df(dldy, y, x):
        n = x.shape[-1] // y.shape[-1]
        dldx = dt.max_unpool_2d(dldy, x, n)
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

def var(x, dim=0):
    return Var.apply(x, axis=dim)

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