import dark
import dark.tensor as dt
import matplotlib.pyplot as plt

def calc_points(range = (-15, 15)):
     x = dt.linspace(range[0], range[1], 50).reshape(1, -1)
     x = dark.Parameter(x)

     fx = dark.add(dark.mul(2, dark.pow(x, 2)), 5) #2*x^2 + 5
     #fx = dark.div(dark.mul(dark.tanh(x), dark.pow(x, 2)), dark.sqrt(x)) #(tanh(x) * x^2) / sqrt(x)
     fx.backward() #calc gradient

     return x.data.squeeze(), \
            x.grad.squeeze(), \
            fx.data.squeeze()


x, x_grad, y = calc_points()

plt.subplot(2, 1, 1)
plt.plot(x, y, color="green")
plt.title("Func")

plt.subplot(2, 1, 2)
plt.plot(x, x_grad, color="red")
plt.title("Grad")

plt.show()