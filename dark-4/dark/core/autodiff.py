import dark.tensor as dt

class Node():
    op = None
    inputs = []

    data = None
    requires_grad = True
    _grad = None

    def __init__(self, val):
        self.data = dt.asarray(val)
        if len(self.data.shape) == 0: self.data = self.data.reshape((1, 1))

    def backward(self):
        if self.grad is None:
            self.grad = dt.ones_like(self.data)

        order = self._topological_sort()   
        for node in order:
            dldx = node.op.differentiate(node)
            
            for k, pd in enumerate(dldx):
                assert node.inputs[k].data.shape == pd.shape #grad has to be the same shape as the node's output
                node.inputs[k].grad = pd

    def zero_grad(self):
        if self._grad is not None:
            self._grad.fill(0)

        for node in self.inputs:
            node.zero_grad()

    def detach(self):
        n = Constant(self.data.copy())
        n.requires_grad = False
        
        return n

    def _topological_sort(self):
        order = []
        seen = set()

        def visit(var):
            if var in seen:
                return
            
            for m in var.inputs:
                visit(m)
                        
            seen.add(var)     
            if type(var) == Node:
                order.insert(0, var)

        visit(self)
        return order

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, data):
        if self._grad is None:
            self._grad = data 
        else:
            self._grad += data 

    def __repr__(self):
        classname = type(self).__name__
        #opClass = type(self.op).__name__
        arraystring = dt.array2string(self.data, precision=4)
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


class Operation():

    @classmethod
    def apply(op_cls, *inputs, **kwargs):
        inputs = list(inputs)
        for k, n in enumerate(inputs):
            if isinstance(n, Node): continue
            inputs[k] = Constant(n)

        op = op_cls()
        x = [n.data for n in inputs]
        y = op.forward(*x, **kwargs)

        out_node = Node(y)
        out_node.op = op
        out_node.inputs = inputs
        
        return out_node

    def differentiate(self, node):
        dldy = node.grad
        x = [n.data for n in node.inputs]
        y = node.data

        result = self.backward(dldy, y, *x)
        return result

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, grad, out, *inputs):
        raise NotImplementedError()
