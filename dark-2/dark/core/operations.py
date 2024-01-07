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
        
class Reshape(Operation):

    def forward(self, input, **kwargs):
        self.out_shape = kwargs["shape"]  
        out = input.reshape(self.out_shape)

        return out
    
    def backward(self, grad, out, input):
        dldy = grad.reshape(input.shape)

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

def reshape(x, shape):
    return Reshape.apply(x, shape=shape)