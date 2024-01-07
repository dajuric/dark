import dark.tensor as dt
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
            
   X_curve = np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25))
   Y_curve  = eval_func(np.reshape(X_curve, (2, -1)).T)
   Y_curve = Y_curve.reshape(-1, 25)
   Y_curve = dt.numpy(Y_curve)
   
   ax.plot_surface(X_curve[0], X_curve[1], Y_curve, color="red")
   ax.scatter(X[:, 0], X[:, 1], Y)

   plt.draw()
   plt.pause(0.1)
   