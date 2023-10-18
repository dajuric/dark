import numpy as np
import dark

x = dark.Parameter(np.array([[10.0]]))
cA = np.array([[2]])
cB = np.array([[5]])

for _ in range(1000):
     fx = dark.add(dark.mul(2, dark.pow(x, cA)), cB) #2*x^2 + 5
     fx.backward() #calc gradient

     x.value = x.value - 0.1 * x.grad #optimize
     x.zero_grad() #zero grad
     
print(x) #0