# Let's create mini PyTorch from scratch - Dark - Module 1 (2/2)

Previously, we have seen how a function graph is built by an example. We have also seen how backpropagation works as well as optimization on a simple 1D quadratic function minimization.

Now, let us start implementing these concepts. Our own minimal version of PyTorch is going to be called [Dark](https://github.com/dajuric/dark/tree/main/dark-1)! 

The framework will follow principles of auto-differentiation and graph building which is often skip in various neural network builds found on the web. Such approach enables us to have an arbitrary complex functions and we do not have to derive partial derivatives ourselves, but it is done automatically.

In this tutorial we will be constrained to scalars and the support for matrices will follow later. However, the basis will be the same.

## Implementation

Going back to our sample from the last module written in PyTorch, let us write the code in our newly (not yet created - *dark* framework):

```python
import dark

x = dark.Parameter(10)

for _ in range(100):
     fx = dark.add(dark.mul(2, dark.pow(x, 2)), 5) #2*x^2 + 5
     fx.backward() #calc gradient

     x.value = x.value - 0.1 * x.grad #optimize
     x.zero_grad() #zero grad
     
print(x) #0
```

First, we set our parameter: $x$. As the variable $x$ is the only variable that will be optimized, it is wrapped into Parameter object. The reason behind is that we have to trace variable $x$ to build a graph. 

Next, we construct the function using math functions in 'dark' namespace for the same reason.

We can see that the code is nearly identical to our PyTorch version. We do not have overridden operators (for the sake of simplicity), thus we have to call functions explicitly.

### Graph nodes

TODO: image of node hierarchy

1) Node
2) Constant and Parameter
3) Operation

We will have three node types. One for constant values which do not have gradient - *Constant* and one for parameters which are optimized - *Parameter* node. Both node types will be inherited from the base class *Node* which is also used as an internal graph node type - for intermediate function results.


#### Node
Let us start from the base class *Node*. The class encapsulates two elements: operation and input values. It contains its *value* as well as gradient *_grad*.

Besides the values, the node object will have two methods which we have seen and used before: *backward* and *zero_grad*.

Function *backward* will call the most imporant function *_autodiff*. Function *_autodiff* is a recursive function that calls differentiate operation for a node, sets or adds gradient to parent node(s) and repeat the procedure until the root of the graph is reached.

```python
class Node(): #base node type used also as a internal graph node
    op = None #operation
    inputs = [] #input nodes

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
        #recursive function that sets or accumulates gradients depending on a node type
        if self.op is not None:
            dldx = self.op.differentiate(self)
            for k, pd in enumerate(dldx):
                self.inputs[k].grad = pd #set (Node) or accumulate (Parameter)
        
        for node in self.inputs:
            node._autodiff()

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value 
```


#### Constant

Node constant does not have a gradient - therefore we override getter and setter function and set gradient to *None*.

```python
class Constant(Node): #constant node does not have gradient

    @property
    def grad(self):
        return None

    @grad.setter
    def grad(self, value):
        pass
```

#### Parameter

Parameter node is a node type created explicitly by user for parameter(s) that we want to optimize. Initially, its gradient is set to zero and opposed to its base class *Node* it accumulates gradient, while *Node* only sets a gradient.

The gradient accumulation is done because *Parameter* can have multiple child nodes and thus gradient is the sum of all gradients going into that node. [TODO: check that fact]

```python
class Parameter(Node): #parameter node which accumulates gradient

    def __init__(self, val):
        super().__init__(val)
        self._grad = 0

    @Node.grad.setter
    def grad(self, value):
        self._grad += value
```

#### Operation

*Operation* object is a member of a *Node* and it has two important functions: *apply* and *differentiate*.

Function *apply* creates a Node, applies an operation and connects node's inputs to a graph being built in such fashion. The end result is built graph. The function is called explicitly by a user.

Function *differentiate* calculates derivative for the node using input(s) *x*, nodes value *y* calculated in the forward pass and a gradient propagated from its child(ren) *dldy*. The function is called by its associated node.

```python
is_training = True #remember the input values only if we are in training mode

class Operation():

    @classmethod
    def apply(op, *inputs, **kwargs):
        #propagate calculated values forward

        #convert numbers into constant node type
        inputs = list(inputs)
        for k, n in enumerate(inputs):
            if isinstance(n, Node): continue
            inputs[k] = Constant(n)

        x = [n.value for n in inputs]
        y = op._f(*x, **kwargs)

        out_node = Node(y)
        if is_training: #remember inputs and op only if we are training
            out_node.op = op
            out_node.inputs = inputs
        
        return out_node

    @classmethod
    def differentiate(op, node):
        #get inputs and calculate gradient for parent node(s)

        dldy = node.grad
        x = [n.value for n in node.inputs]
        y = node.value

        result = op._df(dldy, y, *x)
        return result

    @staticmethod
    def _f(x):
        #calc function value
        raise Exception("Derived class has to implement this method.")

    @staticmethod
    def _df(dldy, y, *x):
        #calculate function derivative using child node propagation (dldy), function value (y) and inputs (x)
        raise Exception("Derived class has to implement this method.")
```

##### Operation samples

One concrete class of an *Operation* class is operation power of. *apply* function is wrapped inside of a function for a convenience.

```python
class Pow(Operation):

    @staticmethod
    def _f(x, n):
        return pow(x, n)

    @staticmethod
    def _df(dldy, y, x, n):
        return [n * pow(x, n - 1) * dldy]

def pow(x, n):
    return Pow.apply(x, n)
```

Another sample is multiplication function that has two arguments ($a$ and $b$). From the sample below we can see that function *_df* returns a tuple which is then processed and propagated by its corresponding class *Node*.

```python
class Mul(Operation):

    @staticmethod
    def _f(a, b):
        return a * b

    @staticmethod
    def _df(dldy, y, a, b):
        dlda = dldy * b
        dldb = dldy * a

        return dlda, dldb

def mul(a, b):
    return Mul.apply(a, b)
```

## Remarks and Final Thoughts
All images used in this article are made by the author. 
The entire source code for this tutorial is available [HERE](https://github.com/dajuric/dark/tree/main/dark-1).


I hope you have learned something new and stay tuned for the next tutorial where we will extend our framework to work with matrices.

## References
[1] https://github.com/a-nico/ArrayFlow
[2] https://minitorch.github.io/

