import torch
from torch.nn import *
from config import *
from util import init_weights
import os

class Generator(Module):
    def __init__(self, nz, ngf, nc):
        super().__init__()
        
        self.nn = Sequential(
            # input: Z
            ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0),
            BatchNorm2d(ngf * 8),
            ReLU(),

            #State: (ngf * 8) x 4 x 4
            ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(ngf * 4),
            ReLU(),

            #State: (ngf * 4) x 8 x 8
            ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(ngf * 2),
            ReLU(),

            #State: (ngf * 2) x 16 x 16
            ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(ngf),
            ReLU(),

            #State: (ngf) x 32 x 32
            ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1),

            #State: nc x 64 x 64
        )

    def forward(self, input):
        x = self.nn(input)
        x = torch.tanh(x)
        return x
    

def get_netG():
    net = None
    if os.path.exists(modelD_path):
        net = torch.load(modelD_path) 
    else:
        net = Generator(nz, 64, 3)
        net.apply(init_weights)
        net = net.to(device)

    return net