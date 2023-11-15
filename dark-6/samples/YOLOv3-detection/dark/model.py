import dark
import dark.nn as nn
from config import *

# https://github.com/deepcam-cn/yolov5-face/blob/master/models/common.py

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        #self.act = self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class BlazeBlock(nn.Module):
    def __init__(self, in_channels,out_channels,mid_channels=None,stride=1):
        super(BlazeBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        if stride>1:
            self.use_pool = True
        else:
            self.use_pool = False

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=mid_channels,kernel_size=5,stride=stride,padding=2,groups=in_channels),
            nn.BatchNorm2d(mid_channels),

            nn.Conv2d(in_channels=mid_channels,out_channels=out_channels,kernel_size=1,stride=1),
            nn.BatchNorm2d(out_channels),
        )

        if self.use_pool:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        branch1 = self.branch1(x)
        out = (branch1+self.shortcut(x)) if self.use_pool else (branch1+x)
        return self.relu(out)    
  
class DoubleBlazeBlock(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None,stride=1):
        super(DoubleBlazeBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        if stride > 1:
            self.use_pool = True
        else:
            self.use_pool = False

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=stride,padding=2,groups=in_channels),
            nn.BatchNorm2d(in_channels),

            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm2d(mid_channels),

            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        if self.use_pool:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        branch1 = self.branch1(x)
        out = (branch1 + self.shortcut(x)) if self.use_pool else (branch1 + x)
        return self.relu(out)
    

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class BlazeFace(nn.Module):
    
    def __init__(self, channels=24):
        super(BlazeFace, self).__init__()
  
        self.backboneA = nn.Sequential(
            Conv(3, channels, k=3, s=2, p=1),
            
            BlazeBlock(channels, channels, channels),
            BlazeBlock(channels, channels, channels),
            
            BlazeBlock(channels, channels*2, channels*2, stride=2), # pix=32
            BlazeBlock(channels*2, channels*2, channels*2),
            BlazeBlock(channels*2, channels*2, channels*2),
            
            DoubleBlazeBlock(channels*2, channels*4, channels, stride=2), # pix=16
            DoubleBlazeBlock(channels*4, channels*4, channels),
            DoubleBlazeBlock(channels*4, channels*4, channels),
            
            DoubleBlazeBlock(channels*4, channels*4, channels, stride=2), #pix=8
            DoubleBlazeBlock(channels*4, channels*4, channels),
            DoubleBlazeBlock(channels*4, channels*4, channels),
        )

        self.backboneB = nn.Sequential(
            DoubleBlazeBlock(channels*4, channels*4, channels, stride=2), #pix=4
            DoubleBlazeBlock(channels*4, channels*4, channels),
            DoubleBlazeBlock(channels*4, channels*4, channels),
        )
        
        self.headL = Conv(channels * 4, NUM_ANCHORS * (C + 4 + 1), k=1, s=1)

        self.headM1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.headM2 = Conv(96+96, NUM_ANCHORS * (C + 4 + 1), k=1, s=1)     
                
        self.apply(BlazeFace.initialize)
        
    def forward(self, x):
        #backbone
        xA = self.backboneA(x)
        xB = self.backboneB(xA)

        #heads (large, medium)
        oL = self.headL(xB)

        oM = self.headM1(xB)
        oM = dark.cat([xA, oM], dim=1)
        oM = self.headM2(oM)

        #reshape
        preds = []
        for o in [oL, oM]:
            b, c, h, w = o.shape
            o = dark.reshape(o, (b, NUM_ANCHORS, (C + 4 + 1), h, w))
            o = dark.moveaxis(o, 2, -1)
            preds.append(o)

        return preds
    
    def initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight.data)
            nn.init.constant_(module.bias.data, 0)
            
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight.data, 1)
            nn.init.constant_(module.bias.data, 0)    


if __name__ == "__main__":
    model = BlazeFace()
    x = .randn((8, 3, 128, 128)).to(device)
    out = model(x)

    #assert out[0].shape == (8, 3, S[0], S[0], C + 5)
    #print("Success!")
    
    torch.save(model, "model.pth")