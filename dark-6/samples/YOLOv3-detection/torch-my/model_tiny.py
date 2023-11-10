import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

# https://github.com/yjh0410/YOLO-Nano/

def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ShuffleV2Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

class ShuffleNetV2(nn.Module):
    def __init__(self,
                 model_size='1.0x',
                 out_stages=(2, 3, 4),
                 with_last_conv=False,
                 kernal_size=3):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        self.out_stages = out_stages
        self.with_last_conv = with_last_conv
        self.kernal_size = kernal_size
        if model_size == '0.5x':
            self._stage_out_channels = [24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self._stage_out_channels = [24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self._stage_out_channels = [24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self._stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, self.stage_repeats, self._stage_out_channels[1:]):
            seq = [ShuffleV2Block(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(ShuffleV2Block(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        
        self._initialize_weights()


    def _initialize_weights(self):

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
                
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
                
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        output = []
        for i in range(2, 5):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)

        return tuple(output)


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, leaky=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True) if leaky else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)

class YOLONano(nn.Module):
    def __init__(self):
        super(YOLONano, self).__init__()
        self.stride = [8, 16, 32]

        self.backbone = ShuffleNetV2()
        c3, c4, c5 = 116, 232, 464
        
        # FPN+PAN
        self.conv1x1_0 = Conv(c3, 96, k=1)
        self.conv1x1_1 = Conv(c4, 96, k=1)
        self.conv1x1_2 = Conv(c5, 96, k=1)

        self.smooth_0 = Conv(96, 96, k=3, p=1)
        self.smooth_1 = Conv(96, 96, k=3, p=1)
        self.smooth_2 = Conv(96, 96, k=3, p=1)
        self.smooth_3 = Conv(96, 96, k=3, p=1)

        # det head
        self.head_det_1 = nn.Sequential(
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1),
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1),
            nn.Conv2d(96, NUM_ANCHORS * (1 + C + 4), 1)
        )
        
        self.head_det_2 = nn.Sequential(
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1),
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1),
            nn.Conv2d(96, NUM_ANCHORS * (1 + C + 4), 1)
        )
        
        self.head_det_3 = nn.Sequential(
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1),
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1),
            nn.Conv2d(96, NUM_ANCHORS * (1 + C + 4), 1)
        )

        self.init_bias()

    def init_bias(self):               
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.head_det_1[-1].bias[..., :NUM_ANCHORS], bias_value)
        nn.init.constant_(self.head_det_2[-1].bias[..., :NUM_ANCHORS], bias_value)
        nn.init.constant_(self.head_det_3[-1].bias[..., :NUM_ANCHORS], bias_value)

    def forward(self, x):
        # backbone
        c3, c4, c5 = self.backbone(x)

        p3 = self.conv1x1_0(c3)
        p4 = self.conv1x1_1(c4)
        p5 = self.conv1x1_2(c5)

        # FPN
        p4 = self.smooth_0(p4 + F.interpolate(p5, scale_factor=2.0))
        p3 = self.smooth_1(p3 + F.interpolate(p4, scale_factor=2.0))

        # PAN
        p4 = self.smooth_2(p4 + F.interpolate(p3, scale_factor=0.5))
        p5 = self.smooth_3(p5 + F.interpolate(p4, scale_factor=0.5))

        # det head
        pred_s = self.head_det_1(p3)
        pred_m = self.head_det_2(p4)
        pred_l = self.head_det_3(p5)

        preds = []
        for pred in [pred_l, pred_m, pred_s]:
            b, c, h, w = pred.shape
            pred = torch.reshape(pred, (b, NUM_ANCHORS, (C + 4 + 1), h, w))
            pred = torch.moveaxis(pred, 2, -1)
            preds.append(pred)
        
        return preds
    

if __name__ == "__main__":
    model = YOLONano().to(device)
    x = torch.randn((8, 3, 320, 320)).to(device)
    out = model(x)

    assert out[0].shape == (8, 3, S[0], S[0], C + 5)
    assert out[1].shape == (8, 3, S[1], S[1], C + 5)
    assert out[2].shape == (8, 3, S[2], S[2], C + 5)
    print("Success!")
    
    torch.save(model, "model.pth")