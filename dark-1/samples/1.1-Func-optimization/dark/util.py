import numpy as np
import matplotlib.pyplot as plt

ax = None

def plot(eval_func, range, x_param): 
   global ax
   if ax is None:
       plt.ion()
       plt.show()  
                   
   X_curve = np.linspace(range[0], range[1], 100)
   Y_curve = [eval_func(x) for x in X_curve]
   
   plt.plot(X_curve, Y_curve, color="red")
   plt.plot(x_param, eval_func(x_param), 'bo')

   plt.draw()
   plt.pause(0.5)
   