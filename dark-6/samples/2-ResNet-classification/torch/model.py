import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False, pool_no=2):
        super().__init__()

        list = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
            #nn.BatchNorm2d(out_channels), 
            nn.ReLU(),
        ]

        if pool: 
            list.append(nn.MaxPool2d(pool_no))

        self.layers = nn.Sequential(*list)

    def forward(self, x):
        return self.layers(x)

class MyConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.conv1 = ConvBlock(1, 64)
        self.conv2 = ConvBlock(64, 128, pool=True, pool_no=4)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True, pool_no=4)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.Flatten(), 
                                        nn.Linear(512, num_classes))
                
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = torch.add(self.res1(out), out)

        out = self.conv3(out)
        out = self.conv4(out)
        out = torch.add(self.res2(out), out)

        out = self.classifier(out)
        return out
