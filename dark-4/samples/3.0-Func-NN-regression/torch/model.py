import torch
import torch.nn as nn

class MyNNBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

class MyNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = MyNNBlock(2, 8)
        self.block2 = MyNNBlock(8, 16)
        self.block3 = MyNNBlock(16, 8)
        self.linear = nn.Linear(8, 1)

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.linear(x)
        return x
    

def get_net():
    net = MyNN()

    return net