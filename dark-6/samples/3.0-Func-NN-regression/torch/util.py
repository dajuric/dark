import torch
import numpy as np
import matplotlib.pyplot as plt

ax = None

def plot(X, Y, eval_func): 
   global ax
   if ax is None:
       plt.ion()
       plt.show()  
       ax = plt.figure().add_subplot(projection='3d')
            
   ax.cla()
            
   X_curve = torch.meshgrid(torch.linspace(0, 1, 25), torch.linspace(0, 1, 25))
   Y_curve  = eval_func(torch.reshape(torch.stack(X_curve), (2, -1)).T)
   Y_curve = Y_curve.reshape(-1, 25)
   
   ax.plot_surface(X_curve[0].numpy(), X_curve[1].numpy(), Y_curve.numpy(), color="red")
   ax.scatter(X[:, 0].numpy(), X[:, 1].numpy(), Y.numpy())

   plt.draw()
   plt.pause(0.1)
   