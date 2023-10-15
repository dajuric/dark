from .autodiff import Operation
from .utils import *
from dark.tensor import *

class AbsoluteValue(Operation):

    @staticmethod
    def _f(x):
        return cp.abs(x)

    @staticmethod
    def _df(dldy, y, x):
        gz = (x > 0)
        lz = cp.logical_not(gz)
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
        dldb = -dldy * a / cp.square(b)

        return reduce_sum(dlda, a.shape), reduce_sum(dldb, b.shape)

class Exp(Operation):

    @staticmethod
    def _f(x):
        return cp.exp(x)

    @staticmethod
    def _df(dydl, y, x):
        return [y * dydl]

class Logarithm(Operation):

    @staticmethod
    def _f(x):
        return cp.log(x)

    @staticmethod
    def _df(dldy, y, x):
        return [dldy / x]

class MatMul(Operation):

    @staticmethod
    def _f(a, b):
        y = cp.matmul(a, b)
        return y

    @staticmethod
    def _df(dldy, y, a, b):
        dlda = cp.matmul(dldy, b.T)
        dldb = cp.matmul(a.T, dldy)
        return dlda, dldb

class Max(Operation):

    @staticmethod
    def _f(a, b):
        return cp.maximum(a, b)

    @staticmethod
    def _df(dldy, y, a, b):
        c = a > b
        dlda = dldy * c
        dldb = dldy * cp.logical_not(c)
        return dlda, dldb

class Mean(Operation):

    @staticmethod
    def _f(x, **kwargs):
        return cp.mean(x, **kwargs, keepdims=True)

    @staticmethod
    def _df(dldy, y, x):
        return [dldy * cp.ones(x.shape) / x.size]

class Min(Operation):

    @staticmethod
    def _f(a, b):
        return cp.minimum(a, b)

    @staticmethod
    def _df(dldy, y, a, b):
        c = a < b
        dlda = dldy * c
        dldb = dldy * cp.logical_not(c)
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
        return cp.power(x, n)

    @staticmethod
    def _df(dldy, y, x, n):
        return [n * cp.power(x, n - 1) * dldy]

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
        return cp.sum(x, **kwargs, keepdims=True)

    @staticmethod
    def _df(dldy, y, x):
        return [dldy * cp.ones(x.shape)]

class SquareRoot(Operation):

    @staticmethod
    def _f(x):
        return cp.sqrt(x)

    @staticmethod
    def _df(dldy, y, x):
        return [.5 * dldy / y]

class View(Operation):

    @staticmethod
    def _f(x, **kwargs):
        outShape = kwargs["shape"]
        return cp.reshape(x, outShape)

    @staticmethod
    def _df(dldy, y, x):
        origShape = x.shape
        return [cp.reshape(dldy, origShape)]


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
