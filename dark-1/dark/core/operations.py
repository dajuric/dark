import numpy as np
from .autodiff import Operation

class AbsoluteValue(Operation):

    @staticmethod
    def _f(x):
        return np.abs(x)

    @staticmethod
    def _df(dldy, y, x):
        gz = (x > 0)
        lz = ~gz
        return [dldy * gz - dldy * lz]

class Add(Operation):

    @staticmethod
    def _f(a, b):
        return a + b

    @staticmethod
    def _df(dldy, y, a, b):
        return dldy, dldy

class Divide(Operation):

    @staticmethod
    def _f(a, b):
        return a / b

    @staticmethod
    def _df(dldy, y, a, b):
        dlda = dldy / b
        dldb = -dldy * a / (b ** 2)

        return dlda, dldb

class Exp(Operation):

    @staticmethod
    def _f(x):
        return np.exp(x)

    @staticmethod
    def _df(dydl, y, x):
        return [y * dydl]

class Logarithm(Operation):

    @staticmethod
    def _f(x):
        return np.log(x)

    @staticmethod
    def _df(dldy, y, x):
        return [dldy / x]

class Max(Operation):

    @staticmethod
    def _f(a, b):
        return np.maximum(a, b)

    @staticmethod
    def _df(dldy, y, a, b):
        c = a > b
        dlda = dldy * c
        dldb = dldy * ~c
        return dlda, dldb

class Min(Operation):

    @staticmethod
    def _f(a, b):
        return np.minimum(a, b)

    @staticmethod
    def _df(dldy, y, a, b):
        c = a < b
        dlda = dldy * c
        dldb = dldy * ~c
        return dlda, dldb

class Mul(Operation):

    @staticmethod
    def _f(a, b):
        return a * b

    @staticmethod
    def _df(dldy, y, a, b):
        dlda = dldy * b
        dldb = dldy * a

        return dlda, dldb

class Pow(Operation):

    @staticmethod
    def _f(x, n):
        return np.power(x, n)

    @staticmethod
    def _df(dldy, y, x, n):
        return [n * np.power(x, n - 1) * dldy]

class Subtract(Operation):

    @staticmethod
    def _f(a, b):
        return a - b

    @staticmethod
    def _df(dldy, y, a, b):
        return +dldy, -dldy

class SquareRoot(Operation):

    @staticmethod
    def _f(x):
        return np.sqrt(x)

    @staticmethod
    def _df(dldy, y, x):
        return [.5 * dldy / y]


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

def max(a, b):
    return Max.apply(a, b)

def min(a, b):
    return Min.apply(a, b)

def pow(x, n):
    return Pow.apply(x, n)

def subtract(a, b):
    return Subtract.apply(a, b)

def sqrt(x):
    return SquareRoot.apply(x)

def mul(a, b):
    return Mul.apply(a, b)
