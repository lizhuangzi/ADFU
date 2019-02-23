import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models.densenet import DenseNet
from torch.utils.checkpoint import checkpoint_sequential

alambda = 0.5

class ResduialDeconv(nn.Module):
    def __init__(self,scale=4):
        super(ResduialDeconv, self).__init__()
        self.scale = scale

        self.upsamp1 = nn.Upsample(scale_factor=2)
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False)
        self.prelu1 = nn.PReLU()

        self.upsamp2 = nn.Upsample(scale_factor=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False)
        self.prelu2 = nn.PReLU()

    def forward(self,x):
        upx1 = self.upsamp1(x)
        x = self.deconv1(x)
        x = self.prelu1(x)
        x = x + upx1

        upx2 = self.upsamp2(x)
        x = self.deconv2(x)
        x = self.prelu2(x)
        x = x + upx2

        return x


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, weightcount=1):
        super(_DenseLayer, self).__init__()
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),

        self.weightcount = weightcount
        if weightcount != 0:
            w = torch.ones(weightcount,1,1,1,1)
            self.weighted = nn.Parameter(w)

    def forward(self, x):
        if self.weightcount == 0:
            new_features = super(_DenseLayer, self).forward(x)
            new_features = torch.unsqueeze(new_features, dim=1)
            return new_features

        else:
            b,g,c,w,h = x .size()
            xin = x.permute(1, 2, 3, 4,0)
            # print torch.squeeze(self.weighted)
            xin = self.weighted * xin
            xin = xin.permute(4, 0, 1, 2,3)
            xin = xin.view(b,g*c,w,h).contiguous()

            new_features = super(_DenseLayer, self).forward(xin)
            new_features = torch.unsqueeze(new_features,dim=1)

            return torch.cat([x, new_features], 1)
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()

        self.map = nn.Sequential(#nn.BatchNorm2d(num_input_features),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(num_input_features, num_output_features,
                                kernel_size=1, stride=1, bias=False)
                      )

        self.Capattention = nn.Sequential(
            nn.Conv2d(num_output_features, 2 * num_output_features, 3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(2 * num_output_features),
            nn.ReLU(inplace=True),

            nn.Conv2d(2 * num_output_features, num_output_features, 3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(num_output_features),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_output_features, num_output_features, 1, stride=1, padding=0, bias=False),
        )

    def forward(self,prevousinput, x):
        b, g, c, w, h = x.size()

        x = x.view(b,g*c,w,h)
        x = self.map(x)
        atten = torch.abs(x - prevousinput)
        atten = self.Capattention(atten)
        atten = F.tanh(atten)
        atten = 0.5 * (x * atten)
        x = atten + x
        return x
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            if i ==0 :
                layer = _DenseLayer(num_input_features, growth_rate, bn_size, weightcount=0)
            else:
                layer = _DenseLayer(i * growth_rate, growth_rate, bn_size,weightcount=i)

            self.add_module('denselayer%d' % (i + 1), layer)
class SpatialAttention(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(SpatialAttention, self).__init__()

        self.bottleneck = nn.Sequential(
                      nn.ReLU(inplace=True),
                      nn.Conv2d(num_input_features, num_output_features,
                                kernel_size=1, stride=1, bias=False)
                      )

        self.Capattention = nn.Sequential(
            nn.Conv2d(num_output_features, 2 * num_output_features, 3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(2 * num_output_features, num_output_features, 3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_output_features, num_output_features, 1, stride=1, padding=0, bias=False),
        )

    def forward(self,prevousinput, x):
        b, g, c, w, h = x.size()

        x = x.view(b,g*c,w,h)
        x = self.bottleneck(x)
        atten = torch.abs(x - prevousinput)
        atten = self.Capattention(atten)
        atten = F.tanh(atten)
        atten = alambda * (x * atten)
        x = atten + x
        return x
class WDB(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super(WDB, self).__init__()
        for i in range(num_layers):
            if i ==0 :
                layer = _DenseLayer(num_input_features, growth_rate, bn_size, weightcount=0)
            else:
                layer = _DenseLayer(i * growth_rate, growth_rate, bn_size,weightcount=i)

            self.add_module('denselayer%d' % (i + 1), layer)
class DenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6,12,48,32),
                 num_init_features=32, bn_size=2,
                ):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('relu0', nn.ReLU()),
        ]))

        num_features = num_init_features

        # WDB1
        self.block1 = WDB(num_layers=block_config[0],
                        num_input_features=num_features,
                        bn_size=bn_size, growth_rate=growth_rate)
        num_features = block_config[0] * growth_rate
        # SA1
        self.SA1 = SpatialAttention(num_input_features=num_features,
                                    num_output_features=growth_rate)

        num_outfeatures = growth_rate + num_init_features

        INP = num_outfeatures

        # WDB2
        self.block2 = WDB(num_layers=block_config[1],
                        num_input_features=INP,
                        bn_size=bn_size, growth_rate=growth_rate)

        num_features = block_config[1] * growth_rate
        # SA2
        self.SA2 = SpatialAttention(num_input_features=num_features,
                                    num_output_features=INP)
        num_outfeatures = INP + INP

        INP = num_outfeatures
        # WDB3
        self.block3 = WDB(num_layers=block_config[2],
                        num_input_features=INP,
                        bn_size=bn_size, growth_rate=growth_rate)

        num_features = block_config[2] * growth_rate
        # SA3
        self.SA3 = SpatialAttention(num_input_features=num_features,
                                    num_output_features=INP)

        num_outfeatures = INP + INP

        INP = num_outfeatures
        # WDB4
        self.block4 = WDB(num_layers=block_config[3],
                        num_input_features=INP,
                        bn_size=bn_size, growth_rate=growth_rate)

        num_features = block_config[3] * growth_rate
        # SA4
        self.SA4 = SpatialAttention(num_input_features=num_features,
                                    num_output_features=INP)

        num_outfeatures = INP + INP

        # final bottleneck
        self.bottleneck = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_outfeatures, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        )

        # residual deconv
        self.deconv = ResduialDeconv()

        # reconstruction
        self.reconstruction = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        primary = self.features(x)

        pre_input = primary

        b1 = self.block1(primary)
        t1 = self.trans1(pre_input, b1)

        concat = torch.cat([pre_input, t1], 1)

        pre_input1 = concat
        b2 = self.block2(concat)
        t2 = self.trans2(pre_input1, b2)

        concat = torch.cat([pre_input1, t2], 1)

        pre_input2 = concat
        b3 = self.block3(concat)
        t3 = self.trans3(pre_input2, b3)

        concat = torch.cat([pre_input2, t3], 1)

        pre_input3 = concat
        b4 = self.block4(concat)
        t4 = self.trans4(pre_input3, b4)

        concat = torch.cat([pre_input3, t4], 1)

        bottleneck = self.bottleneck(concat)
        de = self.deconv(bottleneck)

        out = self.reconstruction(de)
        return out

