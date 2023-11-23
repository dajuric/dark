import torch
import torch.nn as nn
import os
from config import *

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        features = [32, 64, 128, 256]

        # down
        self.down_blocks = nn.ModuleList()
        self.down_reducs = nn.ModuleList()
        for f in features:
            self.down_blocks.append(self.conv_block(in_channels, f))
            self.down_reducs.append(nn.MaxPool2d(2))
            in_channels = f

        # bottleneck
        f = features[-1]
        self.bottleneck = self.conv_block(f, f * 2)

        #up
        self.up_blocks = nn.ModuleList()
        self.up_expnds = nn.ModuleList()
        for f in reversed(features):
            self.up_blocks.append(self.conv_block(2 * f, f))
            self.up_expnds.append(nn.ConvTranspose2d(2 * f, f, kernel_size=2, stride=2))
        
        f = features[0]
        self.out = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # down
        skip_connections = []
        for mb, mr in zip(self.down_blocks, self.down_reducs):
            x = mb(x)
            skip_connections.append(x)
            x = mr(x)

        # bottleneck
        x = self.bottleneck(x)

        # up
        for mb, me, sc in zip(self.up_blocks, self.up_expnds, reversed(skip_connections)):
            x = me(x)
            x = torch.cat([sc, x], dim=1)
            x = mb(x)

        x = self.out(x)
        return x
    
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
                            )
        
        return block

def get_net():
    net = None
    if os.path.exists(model_path):
        net = torch.load(model_path) 
    else:
        net = UNet(3, 1)
        net = net.to(device)

    return net


if __name__ == "__main__":
    input = torch.rand((4, 3, 64, 64))
    net = UNet(3, 1)

    out = net(input)
    print(out.shape)