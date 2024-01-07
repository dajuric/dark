from .autodiff import Operation
import math as m

# ------ scalar ------
class AbsoluteValue(Operation):

    def forward(self, x):
        return x if x >= 0 else -x

    def backward(self, dldy, y, x):
        gz = (x > 0)
        lz = ~gz
        return [dldy * gz - dldy * lz]

class Add(Operation):

    def forward(self, a, b):
        return a + b

    def backward(self, dldy, y, a, b):
        return dldy, dldy

class Subtract(Operation):

    def forward(self, a, b):
        return a - b

    def backward(self, dldy, y, a, b):
        return dldy, -dldy

class Mul(Operation):

    def forward(self, a, b):
        return a * b

    def backward(self, dldy, y, a, b):
        dlda = dldy * b
        dldb = dldy * a

        return dlda, dldb

class Divide(Operation):

    def forward(self, a, b):
        return a / b

    def backward(self, dldy, y, a, b):
        dlda = dldy / b
        dldb = -dldy * a / (b ** 2)

        return dlda, dldb

class Exp(Operation):

    def forward(self, x):
        return m.exp(x)

    def backward(self, dydl, y, x):
        return [y * dydl]

class Logarithm(Operation):

    def forward(self, x):
        return m.log(x)

    def backward(self, dldy, y, x):
        return [dldy / x]
    
class Tanh(Operation):
    
    def forward(self, x):
        return m.tanh(x)
    
    def backward(self, dldy, y, x):
        return [dldy * (1 - y * y)]

class Max(Operation):

    def forward(self, a, b):
        return a if a > b else b

    def backward(self, dldy, y, a, b):
        c = a > b
        dlda = dldy * c
        dldb = dldy * (~c)
        
        return dlda, dldb

class Min(Operation):

    def forward(self, a, b):
        return a if a < b else b

    def backward(self, dldy, y, a, b):
        c = a < b
        dlda = dldy * c
        dldb = dldy * (~c)
        return dlda, dldb

class Pow(Operation):

    def forward(self, x, n):
        return m.pow(x, n)

    def backward(self, dldy, y, x, n):
        return [n * m.pow(x, n - 1) * dldy]

class SquareRoot(Operation):

    def forward(self, x):
        return m.sqrt(x)

    def backward(self, dldy, y, x):
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

def tanh(x):
    return Tanh.apply(x)

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

def neg(x):
    return Subtract.apply(0, x)