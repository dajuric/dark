import torch
import torch.nn as nn
import os
from config import *

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

        self.flatten = nn.Flatten()
        self.block1 = MyNNBlock(IM_SIZE*IM_SIZE, 512)
        self.block2 = MyNNBlock(512, 512)
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        x = self.flatten(x)

        x = self.block1(x)
        x = self.block2(x)

        x = self.linear(x)
        return x
    

def get_net():
    net = None
    if os.path.exists(model_path):
        net = torch.load(model_path) 
    else:
        net = MyNN()

    net = net.to(device)
    return net