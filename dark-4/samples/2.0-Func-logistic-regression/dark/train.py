import numpy as np
import dark
from dark import Parameter
import matplotlib.pyplot as plt
from util import *
from dataset import *

def sigmoid(x):
    den = dark.add(1, dark.exp(dark.neg(x)))
    return dark.div(1, den)

def bce_loss(predictions, targets):
    lossA = dark.mul(targets, dark.log(predictions))
    lossB = dark.mul(dark.subtract(1, targets), dark.log(dark.subtract(1, predictions)))

    loss  = dark.mean(dark.add(lossA, lossB), dim=0)
    loss  = dark.neg(loss)

    return loss 

def eval_model(X, weights, bias):
   Y  = dark.add(dark.matmul(X, weights), bias) # x * w + b
   Y = sigmoid(Y)
   return Y


X, y = get_data()
weights = Parameter(np.random.randn(2, 1))
bias    = Parameter(np.random.randn(1, 1))

for _ in range(500):
   estimates  = eval_model(X, weights, bias)

   if np.any(estimates.data < 1e-10): break
   loss = bce_loss(estimates, y) #numerically unstable if prediction is close to zero !!!
   loss.backward()

   for p in [weights, bias]:
      newVal = p.data - 0.1 * p.grad
      p.data = newVal
      
   print(loss)
   plot(X, y, lambda x: eval_model(x, weights, bias).data)

plt.show(block=True)