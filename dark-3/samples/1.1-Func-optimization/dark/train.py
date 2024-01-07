import dark
from util import *

def eval_func(x):
     fx = dark.add(dark.mul(2, dark.pow(x, 2)), 5) #2*x^2 + 5
     return fx


x = dark.Parameter(10.0)

for _ in range(15):
     print(f"Value of x is: {x.data.item():2.1f}")
     fx = eval_func(x)
     fx.backward() #calc gradient

     plot(lambda x: eval_func(x).data, (-15, 15), x.data)

     x.data = x.data - 0.1 * x.grad #optimize
     x.zero_grad() #zero grad
     
plt.show(block=True)