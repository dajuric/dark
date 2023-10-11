# Module 2

In this tutorial we will extend our framework to work with multi dimensional arrays (tensors) and make a basic sample: logistic regression [TODO].

In order to add support for tensors two things have to be modified:
1) Node types have to support tensor support
2) Operations have to support tensors

Only modifications will be shown - the rest of the code is the same as before and therefore it will be skipped (...).

## Node modifications
The most obvious modification in *Node* is that all values and gradients are NumPy arrays, even the values which are converted into [1, 1] arrays.
Second most important check is that all gradients must have the same shape as as their corresponding values in order to be propagated correctly - *_audtodiff* function.

```python
class Node():
    op = None
    inputs = []

    value = None
    _grad = None

    def __init__(self, val):
        self.value = np.asarray(val)
        if len(self.value.shape) == 0: self.value = self.value.reshape((1, 1))

    def backward(self):
        ...

    def zero_grad(self):
        ...

    def _autodiff(self):
        if self.op is not None:
            dldx = self.op.differentiate(self)
            for k, pd in enumerate(dldx):
                assert self.inputs[k].value.shape == pd.shape #grad has to be the same shape as the node's output
                self.inputs[k].grad = pd #set or add
        
        ...
```

#### Parameter

*Parameter* node accumulates all backpropagated values to its gradient - nothing new here. 
When dealing with tensors, we reduce incoming derivative by summation so it can be added to out gradient. [TODO: example why]

```python
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
```

#### Operations

An *Operation* class stays the same. However, some concrete operations have one extension/postprocessing method: *reduce_sum* which is necessary when dealing with matrices because gradient has to have the same dimensions as its corresponding input value. 

```python
class Add(Operation):

    @staticmethod
    def _f(a, b):
        return a + b

    @staticmethod
    def _df(dldy, y, a, b):
        return reduce_sum(dldy, a.shape), reduce_sum(dldy, b.shape)
```

Let us look that by example:
[TODO example]

So we can finnally write sum-reduction function which does the following.
It searches for a dimension in tensor which is the least similar (larger) compared to target dimension.
Next, it reduces that dimension by summing all elements along its axis.
The procedure is repeated until all dimensions do not correspond to target dimensions.

```python
def reduce_sum(tensor, targetShape):
    assert len(tensor.shape) == len(targetShape)

    result = tensor.copy()
    while True:
        srcShape = result.shape
        
        reduceDim = np.argmin(np.array(targetShape) - np.array(srcShape))
        if srcShape[reduceDim] == targetShape[reduceDim]:
            return result

        result = np.sum(result, axis=reduceDim, keepdims=True)
```

---------------------
---------------------

## Sample - Logistic Regression

Let us build logistic regression - neural network with a single neuron for binary classification.
If you want to know more about the logistic regression please take a look at: [TODO]. 

[TODO: image of input data]

Since we have all the primitvie functions, their derivatives and auto-diff mechanism, creating logistic regression consists of creating sigmoid function, binary cross entropy (BCE) loss and a single neuron.

```python
def bce_loss(predictions, targets):
    ... #try to do it yourself (or look at the source code)

def sigmoid(x):
    den = dark.add(1, dark.exp(dark.neg(x)))
    return dark.div(1, den)


X, y = get_data() #function that outputs random 2D points while y are the labels (0, 1)

weights = Parameter(np.random.randn(2, 1))
bias    = Parameter(np.random.randn(1, 1))
```

Finally, our optimization procedure consists of a evaluating forward pass: $f(x) = \sigma(WX + b)$, calculating BCE loss and finally calculating gradients and updating the parameters $W$ and $b$. Because calculating BCE alone is unstable when predictions are close to zero we stop if any estimates are close to zero.
Because of that PyTorch framework combines BCE with sigmoid and cross entropy loss with softmax.

```python
for _ in range(300):
   estimates  = dark.add(dark.matmul(X, weights), bias) # x * w + b
   estimates = sigmoid(estimates)

   if np.any(estimates.value < 1e-4): break
   loss = bce_loss(estimates, y) #numerically unstable if prediction is close to zero !!!

   loss.backward()
   print(loss)

   for p in [weights, bias]:
      newVal = p.value - 0.03 * p.grad
      p.value = newVal
```

## Sample - Simple NN
TODO - image + code change

## Remarks and Final Thoughts
We have seen in this tutorial how to extend our *dark* framework to work with tensors and finally how to make logistic regression from scratch.
In the next tutorial, we will extend out framework to include modules, Linear NN layer and loss functions just like in PyTorch.

## References
[1]    
[2] https://minitorch.github.io/

