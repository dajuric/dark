import dark

x = dark.Parameter(10)

for _ in range(15):
     print(f"Value of x is: {x.value:2.1f}")

     fx = dark.add(dark.mul(2, dark.pow(x, 2)), 5) #2*x^2 + 5
     fx.backward() #calc gradient

     x.value = x.value - 0.1 * x.grad #optimize
     x.zero_grad() #zero grad