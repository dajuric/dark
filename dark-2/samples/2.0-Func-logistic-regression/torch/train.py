import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import torch
import matplotlib.pyplot as plt
from util import *
from dataset import *

def sigmoid(x):
    den = torch.add(1, torch.exp(torch.neg(x)))
    return torch.div(1, den)

def bce_loss(predictions, targets):
    lossA = torch.mul(targets, torch.log(predictions))
    lossB = torch.mul(torch.subtract(1, targets), torch.log(torch.subtract(1, predictions)))

    loss  = torch.mean(torch.add(lossA, lossB), dim=0)
    loss  = torch.neg(loss)

    return loss 

def eval_model(X, weights, bias):
   Y  = torch.add(torch.matmul(X, weights), bias) # x * w + b
   Y = sigmoid(Y)
   return Y


X, y = get_data()
weights = torch.tensor(np.random.randn(2, 1), requires_grad=True)
bias    = torch.tensor(np.random.randn(1, 1), requires_grad=True)

for _ in range(500):
   estimates  = eval_model(X, weights, bias)

   if torch.any(estimates.data < 1e-10): break
   loss = bce_loss(estimates, y) #numerically unstable if prediction is close to zero !!!
   loss.backward()

   for p in [weights, bias]:
      newVal = p.data - 0.1 * p.grad
      p.data = newVal
      
   print(loss)
   plot(X, y, lambda x: eval_model(torch.tensor(x), weights, bias).data)

plt.show(block=True)