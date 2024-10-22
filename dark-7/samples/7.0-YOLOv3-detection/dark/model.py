import dark
import dark.nn as nn
from dark.nn.init import default_init_weights
from config import *

# https://github.com/deepcam-cn/yolov5-face/blob/master/models/common.py
# https://vincentblog.xyz/posts/face-detection-for-low-end-hardware-using-the-blaze-face-architecture

class Conv(nn.Module):
    def __init__(self, c_in, c_out, k, s):
        super().__init__()
        
        self.conv = nn.Conv2d(c_in, c_out, k, stride = s, padding = k // 2)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class BlazeBlock(nn.Module):
    def __init__(self, c_in, c_out, c_mid = None):
        super().__init__()
        c_mid = c_mid or c_in

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=c_in,out_channels=c_mid,kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(c_mid),

            nn.Conv2d(in_channels=c_mid, out_channels=c_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(c_out),
        )

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out_block = self.block(x)
        out = out_block 

        return self.relu(out)    
  
class BlazeSkipBlock(nn.Module):
    def __init__(self, c_in, c_out, c_mid = None):
        super().__init__()
        c_mid = c_mid or c_in

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_mid, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(c_mid),

            nn.Conv2d(in_channels=c_mid, out_channels=c_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(c_out),
        )

        self.shortcut = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(c_out),
        )

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        block = self.block(x)
        out = dark.add(block, self.shortcut(x))

        return self.relu(out) 

class BlazeFace(nn.Module):
    def __init__(self):
        super().__init__()
  
        self.backboneM = nn.Sequential(
            Conv(3, 24, k=3, s=2),
            
            BlazeBlock(24, 24),
            BlazeBlock(24, 28),
            
            BlazeSkipBlock(28, 32), # pix=32
            BlazeBlock(32, 36),
            BlazeBlock(36, 42),
            
            BlazeSkipBlock(42, 48), # pix=16
            BlazeBlock(48, 56),
            BlazeBlock(56, 64),
            BlazeBlock(64, 72),
            BlazeBlock(72, 80),
            BlazeBlock(80, 88),

            BlazeSkipBlock(88, 96), # pix=8
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96)
        )

        self.backboneL = nn.Sequential(
            BlazeSkipBlock(96, 96), # pix=4
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96)
        )

        self.headL = Conv(96, NUM_ANCHORS * (C + 4 + 1), k=1, s=1)
        self.headM = Conv(96, NUM_ANCHORS * (C + 4 + 1), k=1, s=1)     
                
        self.apply(BlazeFace.initialize)
        
    def forward(self, x):
        #backbone
        xM = self.backboneM(x)
        xL = self.backboneL(xM)

        #heads (large, medium)
        oL = self.headL(xL)
        oM = self.headM(xM)

        #reshape
        preds = []
        for o in [oL, oM]:
            b, c, h, w = o.data.shape
            o = dark.reshape(o, (b, NUM_ANCHORS, (C + 4 + 1), h, w))
            o = dark.moveaxis(o, 2, -1)
            preds.append(o)

        return preds
    
    def initialize(module):
        default_init_weights(module)

        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weights)
            module.bias.data[:] = 0


if __name__ == "__main__":
    import dark.tensor as dt

    model = BlazeFace()
    x = dt.random.randn(*(8, 3, 128, 128))
    out = model(x)

    assert out[0].data.shape == (8, 3, S[0], S[0], C + 5)
    assert out[1].data.shape == (8, 3, S[1], S[1], C + 5)
    print("Success!")