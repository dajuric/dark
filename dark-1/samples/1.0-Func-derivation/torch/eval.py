import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
import matplotlib.pyplot as plt

def calc_points(range = (-15, 15)):
     x = torch.linspace(range[0], range[1], 50).reshape(1, -1)
     x = torch.tensor(x, requires_grad=True)

     fx = torch.add(torch.mul(2, torch.pow(x, 2)), 5) #2*x^2 + 5
     #fx = torch.div(torch.mul(torch.tanh(x), torch.pow(x, 2)), torch.sqrt(x)) #(tanh(x) * x^2) / sqrt(x)
     fx.sum().backward() #calc gradient

     return x.data.squeeze().numpy(), \
            x.grad.squeeze().numpy(), \
            fx.data.squeeze().numpy()


x, x_grad, y = calc_points()

plt.subplot(2, 1, 1)
plt.plot(x, y, color="green")
plt.title("Func")

plt.subplot(2, 1, 2)
plt.plot(x, x_grad, color="red")
plt.title("Grad")

plt.show()