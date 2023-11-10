import torch
import torch.nn as nn
from config import *

architecture_config = [
    (7, 64, 2, 3), #kernel, filters, stride, padding
    "M", #max pooling 2x2

    (3, 192, 1, 1),
    "M",

    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",

    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",

    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),

    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.nn(x)


class YoloV1(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.darknet = self._create_conv_layers(architecture_config, in_channels)
        self.fcs = self._create_fcs()

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1) #TODO: do we need this ?
        x = self.fcs(x)
        return x

    def _create_conv_layers(self, architecture, in_channels):
        layers = []

        for x in architecture:
            if type(x) == tuple:
                k, f, s, p = x
                block = CNNBlock(in_channels, f, kernel_size=k, stride=s, padding=p)
                layers += [block]
                in_channels = f

            elif type(x) == str:
                k, s = 2, 2
                pool = nn.MaxPool2d(kernel_size=2, stride=2)
                layers += [pool]

            elif type(x) == list:
                (kA, fA, sA, pA), (kB, fB, sB, pB), nReps = x

                for _ in range(nReps):
                    blockA = CNNBlock(in_channels, fA, kernel_size=kA, stride=sA, padding=pA)
                    layers += [blockA]
                    in_channels = fA

                    blockB = CNNBlock(in_channels, fB, kernel_size=kB, stride=sB, padding=pB)
                    layers += [blockB]
                    in_channels = fB

        return nn.Sequential(*layers)

    def _create_fcs(self):
        net = nn.Sequential(
            nn.Flatten(),

            nn.Linear(1024 * S * S, 496), #should be 4096
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),

            nn.Linear(496, S * S * (C + 1 * 5))
        )

        return net

            
if __name__ == "__main__":
    imBatch = torch.rand((16, 3, IM_SIZE, IM_SIZE))
    net = YoloV1()

    out = net(imBatch)
    #out = out.reshape(-1, S, S, (1 * 5 + C))
    print(out.shape)
