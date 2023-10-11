import numpy as np
import dark
import dark.nn as nn
from dark.nn.init import default_init_weights
from dark.optim import SGD
from util import MnistDataloader, process_mnist_data

class MyNNBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

class MyNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = MyNNBlock(28*28, 512)
        self.block2 = MyNNBlock(512, 512)
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.linear(x)
        return x

def calc_accuracy(estimates, targets):
    estimates = estimates.value.argmax(1)
    targets = targets.argmax(1)

    size = targets.shape[0]
    correct = (estimates == targets).sum().item()
    return correct / size

(x_train, y_train), _ = MnistDataloader("samples/db-MNIST/").load_data()
x_train, y_train = process_mnist_data(x_train, y_train)

net = MyNN()
net.apply(default_init_weights)

loss_fn = nn.CrossEntropyLoss()
optim = SGD(net.parameters(), lr=0.1)

for _ in range(5):
   estimates  = net(x_train)
   loss = loss_fn(estimates, y_train)
   
   acc =  calc_accuracy(estimates, y_train)
   print(f"Loss: {loss.value.item():2.2f}, acc: {acc:2.2f}")

   optim.zero_grad()
   loss.backward()
   optim.step()