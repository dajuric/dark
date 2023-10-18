import numpy as np
import random

random.seed(0)
np.random.seed(0)


import dark
import dark.nn
from dark import Parameter

def get_data():
    X = []; Y = []

    for i in range(50):
        x_1 = random.random()
        x_2 = random.random()
        y = 1 if x_1 < 0.2 or x_1 > 0.8 else 0 #split
        #y = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0 #xor

        X.append((x_1, x_2))
        Y.append([0, 0]); Y[-1][y] = 1

    X = np.array(X, dtype=np.float32); Y = np.array(Y, dtype=np.float32)
    return X, Y

X, y = get_data()
weights = np.random.randn(2, 2).astype(np.float32)
bias    = np.random.randn(1, 2).astype(np.float32)

dloss_fn = dark.nn.CrossEntropyLoss()
dweights = Parameter(weights)
dbias    = Parameter(bias)

for _ in range(100):
   dresult  = dark.add(dark.matmul(X, dweights), dbias)
   dresult  = dark.max(np.zeros(dresult.value.shape), dresult)

   dloss = dloss_fn(dresult, y)
   dloss.backward()
   print(dloss)

   for p in [dweights, dbias]:
      newVal = p.value - 0.1 * p.grad
      p.value = newVal



import torch
import torch.nn

X = torch.tensor(X, requires_grad=False)
y = torch.tensor(y, requires_grad=False)

tloss_fn = torch.nn.CrossEntropyLoss()
tweights = torch.tensor(weights, requires_grad=True)
tbias    = torch.tensor(bias, requires_grad=True)

for _ in range(100):
   tresult  = torch.add(torch.matmul(X, tweights), tbias)
   tresult  = torch.relu(tresult)
   
   tloss = tloss_fn(tresult, y)
   tloss.backward()
   print(tloss)

   for p in [tweights, tbias]:
      p.data.sub_(0.1 * p.grad)


print()
print(dweights.grad); print(tweights.grad)
print()
print(dbias.grad);    print(tbias.grad)
print("Done")
