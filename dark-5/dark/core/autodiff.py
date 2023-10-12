import numpy as np

is_training = True
precision = np.float64

class Node():
    op = None
    inputs = []

    value = None
    _grad = None

    def __init__(self, val):
        self.value = np.asarray(val, dtype=precision)
        if len(self.value.shape) == 0: self.value = self.value.reshape((1, 1))

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.value)

        self._autodiff()

    def zero_grad(self):
        if self._grad is not None:
            self._grad.fill(0)

        for node in self.inputs:
            node.zero_grad()

    def _autodiff(self):
        if self.op is not None:
            dldx = self.op.differentiate(self)
            for k, pd in enumerate(dldx):
                assert self.inputs[k].value.shape == pd.shape #grad has to be the same shape as the node's output
                self.inputs[k].grad = pd #set or add
        
        #print(f"{self.value.shape} of {self.op}")
        for node in self.inputs:
            node._autodiff()

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value 

    def __repr__(self):
        classname = type(self).__name__
        #opClass = type(self.op).__name__
        arraystring = np.array2string(self.value, precision=4)
        return f"({classname}): {arraystring}"

class Constant(Node):

    @property
    def grad(self) -> None:
        return None

    @grad.setter
    def grad(self, value):
        pass

class Parameter(Node):

    def __init__(self, val):
        super().__init__(val)
        self._grad = np.zeros(self.value.shape, dtype=precision)

    @Node.grad.setter
    def grad(self, value):
        if value.shape[0] > self._grad.shape[0]:
            assert self._grad.shape[0] == 1
            value = np.sum(value, axis=0, keepdims=True)

        self._grad += value

class Operation():

    @classmethod
    def apply(op, *inputs, **kwargs):
        inputs = list(inputs)
        for k, n in enumerate(inputs):
            if isinstance(n, Node): continue
            inputs[k] = Constant(n)

        x = [n.value for n in inputs]
        y = op._f(*x, **kwargs)

        out_node = Node(y)
        if is_training:
            out_node.op = op
            out_node.inputs = inputs
        
        return out_node

    @classmethod
    def differentiate(op, node):
        dldy = node.grad
        x = [n.value for n in node.inputs]
        y = node.value

        result = op._df(dldy, y, *x)
        return result

    @staticmethod
    def _f(x):
        raise NotImplementedError()

    @staticmethod
    def _df(dldy, y, *x):
        raise NotImplementedError()
