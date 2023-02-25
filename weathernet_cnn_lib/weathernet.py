import torch
import torch.hub as hub
import torch.nn as nn
import torch.nn.functional as F
import math

# Inspired by https://github.com/TheCodez/pytorch-LiLaNet
# Implements CNN as described in "Boosting LiDAR-based Semantic Labeling by Cross-Modal Training Data Generation"

pretrained_models = {
    'kitti': {
        'url': '',
        'num_classes': 4
    }
}

def h_calc(conv2d, h_in):
    conv = conv2d.conv
    h_out = math.floor((h_in + 2*conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0]- 1) -1)/conv.stride[0] + 1)
    return h_out

def w_calc(conv2d, w_in):
    conv = conv2d.conv
    w_out = math.floor((w_in + 2*conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1]- 1) -1)/conv.stride[1] + 1)
    return w_out

def lilanet(pretrained=None, num_classes=13):
    """Constructs a LilaNet model.
    
    Args:
        pretrained (string): If not ``None``, returns a pre-trained model. Possible values: ``kitti``.
        num_classes (int): number of output classes. Automatically set to the correct number of classes if ``pretrained`` is specified

    """
    if pretrained is not None:
        model = LiLaNet(pretrained_models[pretrained]['num_classes'])
        model.load_state_dict(hub.load_state_dict_from_url(pretrained_models[pretrained]['url']))
        return model

    model = LiLaNet(num_classes)
    return model

class LiLaNet(nn.Module):
    # LilaNet CNN
    # num_classes: Number of classes to segment input points in to

    def __init__(self, num_classes=2):
        super(LiLaNet, self).__init__()

        # WeatherNet CNN Layers based on LilaBlocks and Dropout
        self.lila1 = LilaBlock(2, 32)
        self.lila2 = LilaBlock(32, 64)
        self.lila3 = LilaBlock(64, 96)
        self.lila4 = LilaBlock(96, 96)
        self.dropout  = nn.Dropout()
        self.lila5 = LilaBlock(96, 64)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

        # Initializing weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # Primary function to create classification based on distance and reflectivity
    def forward(self, x):
        # Assumes x is shape of [batch, channels(distance, reflectivity), height, width]
        x = self.lila1(x)
        x = self.lila2(x)
        x = self.lila3(x)
        x = self.lila4(x)
        x = self.dropout(x)
        x = self.lila5(x)

        x = self.classifier(x)

        return x

class LilaBlock(nn.Module):
    # LilaBlock
    # Modified to fit WeatherNet
    
    def __init__(self, in_channels, n):
        super(LilaBlock, self).__init__()

        self.branch1 = BasicConv2d(in_channels, n, kernel_size=(7,3), padding=(2, 0))
        self.branch2 = BasicConv2d(in_channels, n, kernel_size=3)
        self.branch3 = BasicConv2d(in_channels, n, kernel_size=3, padding=(1,1), dilation=(2, 2))
        self.branch4 = BasicConv2d(in_channels, n, kernel_size=(3,7), padding=(0, 2))
        
        self.conv = BasicConv2d(n * 4, n, kernel_size=1, padding=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        output = torch.cat([branch1, branch2, branch3, branch4], 1)
        output = self.conv(output)
        
        return output

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return F.relu(x, inplace=True)

if __name__ == '__main__':

    # num_classes: Number of classification options available
    # height:
    # width: 
    # model:       Instance of LilaNet CNN to put input data trhough
    # inp:         Input data for testing
    # out:         Output classification

    num_classes, height, width = 2, 64, 1024

    model = LiLaNet(num_classes) # .to('cuda')
    inp   = torch.randn(5, 1, height, width) # .to('cuda')

    out   = model(inp, inp)

    assert out.size() == torch.Size([5, num_classes, height, width])

    print('Pass size check.')