import tortto
import tortto.nn as nn


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
            x = tortto.cat([sc, x], dim=1)
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




class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels, features=[32,64,128,256]):
        super(UNet2, self).__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        for f in features:
            self.down.append(DoubleConv(in_channels, f))
            in_channels = f
        self.bottle_neck = DoubleConv(f, 2 * f)
        for f in reversed(features):
            self.up.append(nn.ConvTranspose2d(2 * f, f, kernel_size=2, stride=2))
            self.up.append(DoubleConv(2 * f, f))
        self.out = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for m in self.down:
            x = m(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottle_neck(x)
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.up), 2):
            skip_connection = skip_connections[i // 2]
            x = self.up[i](x, output_size=skip_connection.shape[-2:])  # upsample
            x = tortto.cat([skip_connections[i // 2], x], dim=1)  # cat
            x = self.up[i + 1](x)  # double conv
        x = self.out(x)
        return x



# adopted from: https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705
class UNet3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 3, 1)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv1 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv3 = self.expand_block(32*2, out_channels, 3, 1)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv1(conv3)
        upconv2 = self.upconv2(tortto.cat([upconv3, conv2], 1))
        upconv1 = self.upconv3(tortto.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                            )
        return expand
    
