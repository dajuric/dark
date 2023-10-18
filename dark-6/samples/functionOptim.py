import dark
import dark.tensor as dt

x = dark.Parameter(dt.array([7]).reshape(1, 1))

for _ in range(40):
     print(f"Value of x is: {x.value}")

     fx = dark.mul(x, 2)
     fx = dark.add(fx, dark.pow(fx, 2))
     #fx = dark.add(dark.pow(fx, 2), fx)
     
     #fx.print_graph()
     #print(fx.topological_sort())
     fx.backward() #calc gradient

     x.value = x.value - 0.01 * x.grad #optimize
     x.zero_grad() #zero grad