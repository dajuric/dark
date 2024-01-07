from torch.nn import *
from config import *
from util import init_weights
import os

class Discriminator(Module):
    def __init__(self, ndf, nc):
        super().__init__()
        
        self.nn = Sequential(
            #input: nc x 64 x 64
            Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1),
            LeakyReLU(0.2),

            #state: ndf x 32 x 32
            Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(ndf * 2),
            LeakyReLU(0.2),

            #state: (ndf x 2) x 16 x 16
            Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(ndf * 4),
            LeakyReLU(0.2),

            #state: (ndf x 4) x 8 x 8
            Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(ndf * 8),
            LeakyReLU(0.2),

            #state: (ndf * 8) x 4 x 4
            Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0),
            Flatten()
        )

    def forward(self, input):
        return self.nn(input)
    

def get_netD():
    net = None
    if os.path.exists(modelD_path):
        net = torch.load(modelD_path) 
    else:
        net = Discriminator(64, 3)
        net.apply(init_weights)
        net = net.to(device)

    return net