import dark
import dark.nn as nn
import dark.tensor as dt

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False, pool_no=2):
        super().__init__()

        list = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(),
        ]

        if pool: 
            list.append(nn.MaxPool2d(pool_no))

        self.layers = nn.Sequential(*list)

    def forward(self, x):
        return self.layers(x)

class Resnet9(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.conv1 = ConvBlock(3, 64)
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
        out = dark.add(self.res1(out), out)

        out = self.conv3(out)
        out = self.conv4(out)
        out = dark.add(self.res2(out), out)

        out = self.classifier(out)
        return out
    

if __name__ == "__main__":
    model = Resnet9()
    
    im = dark.Parameter(dt.random.random((1, 3, 32, 32)))
    result = model(im)
    print(result.data.shape)
