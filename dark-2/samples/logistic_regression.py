import random
import numpy as np
import dark
from dark import Parameter

def get_data():
    X = []; Y = []

    for i in range(50):
        x_1 = random.random()
        x_2 = random.random()

        y = 1 if x_1 < 0.5 else 0 #split

        X.append((x_1, x_2))
        Y.append([y])

    X = np.array(X); Y = np.array(Y)
    return X, Y

def sigmoid(x):
    den = dark.add(1, dark.exp(dark.neg(x)))
    return dark.div(1, den)

def bce_loss(predictions, targets):
    lossA = dark.mul(targets, dark.log(predictions))
    lossB = dark.mul(dark.subtract(1, targets), dark.log(dark.subtract(1, predictions)))

    loss  = dark.mean(dark.add(lossA, lossB), dim=0)
    loss  = dark.neg(loss)

    return loss 


X, y = get_data()

weights = Parameter(np.random.randn(2, 1))
bias    = Parameter(np.random.randn(1, 1))

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