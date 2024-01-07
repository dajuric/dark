import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
from util import *

def eval_func(x):
     fx = torch.add(torch.mul(2, torch.pow(x, 2)), 5) #2*x^2 + 5
     return fx


x = torch.tensor(10.0, requires_grad=True)

for _ in range(15):
     print(f"Value of x is: {x.data.item():2.1f}")
     fx = eval_func(x)
     fx.backward() #calc gradient

     plot(lambda x: eval_func(torch.tensor(x)).data, (-15, 15), x.data)

     x.data = x.data - 0.1 * x.grad #optimize
     x.grad.zero_() #zero grad
     
plt.show(block=True)