import dark
import matplotlib.pyplot as plt

def calc_points(x):
     x = dark.Parameter(x)

     fx = dark.add(dark.mul(2, dark.pow(x, 2)), 5) #2*x^2 + 5
     #fx = dark.div(dark.mul(dark.tanh(x), dark.pow(x, 2)), dark.sqrt(x)) #(tanh(x) * x^2) / sqrt(x)
     fx.backward() #calc gradient

     return x.data, \
            x.grad, \
            fx.data


# we have to calculate gradient for each point because we only support scalars - this example is rewritten in subsequent module
x = []; x_grad = []; y = []
for x_in in range(-15, +15):
     x_pt, x_grad_pt, y_pt = calc_points(x_in)
     x.append(x_pt); x_grad.append(x_grad_pt); y.append(y_pt)

plt.subplot(2, 1, 1)
plt.plot(x, y, color="green")
plt.title("Func")

plt.subplot(2, 1, 2)
plt.plot(x, x_grad, color="red")
plt.title("Grad")

plt.show()