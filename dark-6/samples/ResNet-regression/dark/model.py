import dark
import dark.nn as nn

class Block(nn.Module):
    
    def __init__(self, in_ch, out_ch, stride=1):
        super(Block, self).__init__()
        self.downsample = self.get_identity_downsample(in_ch, out_ch) if stride != 1 else None
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample:
            x = dark.add(x, self.downsample(identity))         

        x = self.relu(x)
        return x
    
    def get_identity_downsample(self, in_channels, out_channels): 
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )
    
    
class Resnet18(nn.Module):
    def __init__(self, image_channels = 3, out_dim = 3):
        super(Resnet18, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        #resnet layers
        self.layer1 = self.make_layer(64, 64, stride=1)
        self.layer2 = self.make_layer(64, 128, stride=2)
        self.layer3 = self.make_layer(128, 256, stride=2)
        self.layer4 = self.make_layer(256, 512, stride=2)
        
        self.finaladjust = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(512, out_dim)
        
    def make_layer(self, in_channels, out_channels, stride):                 
        return nn.Sequential(
            Block(in_channels, out_channels, stride=stride), 
            Block(out_channels, out_channels)
        )
        
    def forward(self, x):  
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.finaladjust(x)
        x = self.fc(x)
        return x 
     

if __name__ == "__main__":
    import dark.tensor as dt
    
    net = Resnet18(3, 8)
    x = dt.random.randn(1, 3, 128, 128)
    res = net(x)
    print(res.data.shape)