import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class BlazeBlock(nn.Module):
    def __init__(self, inp, oup1, oup2, stride=1, kernel_size=5):
        super(BlazeBlock, self).__init__()
        self.stride = stride
        
        self.use_pooling = self.stride != 1
        self.channel_pad = oup2 - inp
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=padding, groups=inp, bias=True),
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, oup1, 1, 1, 0, bias=True),  # piecewise-linear convolution.
            nn.BatchNorm2d(oup1),
            
            nn.ReLU(inplace=True),
            
            # dw
            nn.Conv2d(oup1, oup1, kernel_size=kernel_size, stride=1, padding=padding, groups=oup1, bias=True),
            nn.BatchNorm2d(oup1),
            # pw-linear
            nn.Conv2d(oup1, oup2, 1, 1, 0, bias=True),
            nn.BatchNorm2d(oup2),
        )
        
        if self.use_pooling:
            self.mp = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)
            
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.conv(x)

        if self.use_pooling:
            x = self.mp(x)
            
        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), 'constant', 0)
            
        return self.act(h + x)

class BlazeFace(nn.Module):
    
    def __init__(self, channels=24):
        super(BlazeFace, self).__init__()
  
        self.backbone = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1, bias=True), # pix=64
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            
            BlazeBlock(channels, channels, channels),
            BlazeBlock(channels, channels, channels),
            
            BlazeBlock(channels, channels*2, channels*2, stride=2), # pix=32
            BlazeBlock(channels*2, channels*2, channels*2),
            BlazeBlock(channels*2, channels*2, channels*2),
            
            BlazeBlock(channels*2, channels, channels*4, stride=2), # pix=16
            BlazeBlock(channels*4, channels, channels*4),
            BlazeBlock(channels*4, channels, channels*4)
        )
        
        out_dim = NUM_ANCHORS * (C + 4 + 1)
        self.regressor = nn.Conv2d(channels*4, out_dim, 1, bias=True)
        
        self.apply(BlazeFace.initialize)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        
        b, c, h, w = x.shape
        x = torch.reshape(x, (b, NUM_ANCHORS, (C + 4 + 1), h, w))
        x = torch.moveaxis(x, 2, -1)
        return [x]
    
    def initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight.data)
            nn.init.constant_(module.bias.data, 0)
            
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight.data, 1)
            nn.init.constant_(module.bias.data, 0)


if __name__ == "__main__":
    model = BlazeFace().to(device)
    x = torch.randn((8, 3, 128, 128)).to(device)
    out = model(x)

    assert out[0].shape == (8, 3, S[0], S[0], C + 5)
    print("Success!")
    
    torch.save(model, "model.pth")