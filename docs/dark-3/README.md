# Module 3

So far we have developed the core part of the framework: graph building, primitive functions with derivatives and the auto-diff engine. This creates the foundation for the next step - higher logic which includes *Module* class which is the foundation for all other neural network parts.

Take a look at the code for the logistic regression from the previous tutorial part. There we have defined our neuron through matrices explicitly, we also wrote BCE loss from scratch and an SGD optimizer. Now, let us put those elements into their own units, so we can use them easily.

New code for logistic regression sample using modules which will be implemented in this tutorial is shown below. It looks and feels much more like standard PyTorch code!

```python
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.net(x)


X, y = get_data()
net = LogisticRegression(2)
loss_fn = nn.BCEWithLogits()
optim = optim.SGDOptimizer(net.parameters())

for _ in range(300):
   estimates  = net(X)
   loss = loss_fn(estimates, y)

   loss.backward()
   optim.zero_grad()
   optim.optimize()
   
   print(loss)
```

Even though the code is not shorter for our sample, imagine if we had multiple fully connected NN layers and activation functions: we would have to care about that explicitly by ourselves. That is why PyTorch adds higher level of abstractions through class *Module*, so shall we as well.


## Class Module

Let us build our foundation class - *Module*. The *Module* class encapsulates all parameters (node class *Parameter*) which can be defined across multiple module classes. A basic example would be the following network:

```python
class MyNNBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

class MyNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = MyNNBlock(28*28, 512)
        self.block2 = MyNNBlock(512, 512)
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.linear(x)
        return x
```
We can see that the neural network above consists of three feed forward layers. Two of them have an activation function and are put in different module for the sake of clarity.

*Linear* module encapsulates weights and bias parameters for that layer. 

```python
class ZeroParam(Parameter):
    def __init__(self, *shape):
        val = np.zeros(shape)
        super().__init__(val)

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weights = ZeroParam(in_features, out_features)
        self.bias    = ZeroParam(1, out_features)

    def forward(self, x):
        result = dark.add(dark.matmul(x, self.weights), self.bias)
        return result
```

Our graph is built when forward method of a module (and all submodules) is called. First, MyNN *forward* method is called which calls MyNNBlock *forward* methods which call Linear forward methods which finally adds nodes to a graph by calling dark.* functions as we have seen it before when we wrote logistic regression using *Parameter* type variables explicitly.

Therefore, our *Module* class has to have two abilities:
1) to collect all immediate depending modules
2) to collect all the parameters so we can pass them to an optimizer

For the first case we do not do anything special since *forward* method will automatically call all other *forward* methods in other submodules - thus build a graph.
The second case requires two steps: tracking  when a parameter or a module is assigned to an object and getting all the parameters recursively.

We track a variable assignment by overriding built-in *__setattr__* function and after we get all immediate parameters and modules in that fashion, we can create recursive method *get_params* which will return all the parameters of all submodules including the calling module - as shown below. 

```python
class Module():
    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def forward(self, *x):
        pass

    def __call__(self, *x):
        return self.forward(*x)

    def modules(self):
        return self._modules.values()

    def parameters(self):
        params = []

        def get_params(module):
           for p in module._parameters.values():
               params.append(p)

           for m in module._modules.values():    
               get_params(m)

        get_params(self)
        return params

    def __setattr__(self, key, value):
        if isinstance(value, Parameter):
            self._parameters[key] = value
        elif isinstance(value, Module):
            self._modules[key] = value

        super().__setattr__(key, value)
```

Now, we are ready to write a few classes that inherit our abstract base *Module* class such as already shown *Linear* class, *ReLU* and *BCEWithLogits* by simply overriding *forward* method of the base class.

### ReLU
[TODO]

### Flatten
[TODO]

### BCEWithLogits Loss

As it has been mentioned in the last tutorial our naive implementation of a binary cross entropy loss (BCE) leads to instability for small or large predictions.
Therefore PyTorch framework came up with BCE with logits loss function which is nothing else but BCE + sigmoid combined.

Our best bet is to derive manually this function and use it as a primitive function just like dark.add, dark.sub, etc. Such implementation will not suffer from numerical instability as the naive implementation.

The derivation is already done for us in [1]. We just need to write some code. First, we write our *BCEWithLogits* operation:

```python
class BCEWithLogits(Operation):

    @staticmethod
    def _f(x, y, **kwargs):
        x = sigmoid(x)
        out = y * np.log(x) + (1-y) * np.log(1-x)
        out = -np.mean(out, **kwargs, keepdims=True)
        return out

    @staticmethod
    def _df(dldy, y, x):
        return (sigmoid(x) - y) * x

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
```

Then, we wrap it in a static *dark.bce_with_logits* function.
```python

def bce_with_logits(x, y, dim=0):
    return BCEWithLogits.apply(x, y, axis=dim)
```

Finally, we can write our loss as a module using newly created operation.
```python
class BCEWithLogitsLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, dim=0):
        loss = dark.bce_with_logits(predictions, targets, dim)
        return loss
```

In a similar way, cross entropy loss with logits is also derived [2]. Please check the implementation for details.

## Optimizer

A optimizer class receives all parameters from our neural network (*Module*) via method *module.parameters()*. 
It has two methods as shown bellow. The only method which need to be implemented is a *step()* method. 

```python
class Optimizer():
    def __init__(self, parameters):
        self.parameters = parameters

    def step(self):
        raise NotImplementedError()

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()
```

We have written before a simple implementation of a stohastic gradient descent optimizer. Below is the same implementation, but this time extnded so it works on all parameters and is wrapped into its own class.

```python
class SGD(Optimizer):
    def __init__(self, parameters, lr=1e-3):
        assert lr > 0 and lr <= 1

        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for i, p in enumerate(self.parameters):
            update = -self.lr * p.grad
            p.value = p.value + update
```

In the provided code, a momentum has been added as well, while here omitted for simplicity.

### Initialization
[TODO]

## Sample - multi layered NN

With the implemented *Module*, *Linear* and *ReLU* classes, BCE and Cross-Entropy losses and finally SGD optimizer we are ready to write multi-layered feed forward network.
A sample code for a classifier which operates on 2D data is shown bellow. Implementation for a network is arbitrary and multi modular as shown before in this tutorial - therefore omitted.
This time we can solve much more complex tasks, that logistic regression could not do.

```python
class MyNNBlock(Module):
    ...

class MyNN(Module):
    ...

X, y = get_data()
net = MyNN()
loss_fn = nn.BCEWithLogits()
optim = optim.SGDOptimizer(net.parameters())

for _ in range(10):
   estimates  = net(X)
   loss = loss_fn(estimates, y)

   loss.backward()
   optim.zero_grad()
   optim.optimize()
   
   print(loss)
```

## Remarks and Final Thoughts

We have extended our *dark* framework with base class that encapsulates all parameters into single units - *Module*. Subsequently, we wrote *Linear* layer among others.
Before we can write a multi-class classifier samples with epochs and data loading, some helper classes are missing: *Dataset* and *DataLoader* which will be added in the next tutorial. 

## References   
[1] https://medium.com/@andrewdaviesul/chain-rule-differentiation-log-loss-function-d79f223eae5   
[2] https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1   
[3] https://minitorch.github.io/

