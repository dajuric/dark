is_training = True

class Node():
    op = None
    inputs = []

    value = None
    _grad = None

    def __init__(self, val):
        self.value = val
       
    def backward(self):
        if self.grad is None:
            self.grad = 1

        self._autodiff()

    def zero_grad(self):
        if self._grad is not None:
            self._grad = 0

        for node in self.inputs:
            node.zero_grad()

    def _autodiff(self):
        if self.op is not None:
            dldx = self.op.differentiate(self)
            for k, pd in enumerate(dldx):
                self.inputs[k].grad = pd #set or add
        
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
        return f"({classname}): {self.value}"

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
        self._grad = 0

    @Node.grad.setter
    def grad(self, value):
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
