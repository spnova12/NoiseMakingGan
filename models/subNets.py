import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

def weights_init(m):
    """
    custom weights initialization called on netG and netD
    https://github.com/pytorch/examples/blob/master/dcgan/main.py
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)  # 0.02
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

####################################################################################################################


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    """
    https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch
    """

    def __init__(self, nChannels, nDenselayer, growthRate):
        """
        :param nChannels: input feature 의 channel 수
        :param nDenselayer: RDB(residual dense block) 에서 Conv 의 개수
        :param growthRate: Conv 의 output layer 의 수
        """
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        # local residual 구조
        out = out + x
        return out


def RDB_Blocks(channels, size):
    bundle = []
    for i in range(size):
        bundle.append(RDB(channels, nDenselayer=8, growthRate=64))  # RDB(input channels,
    return nn.Sequential(*bundle)

####################################################################################################################


class ResidualBlock(nn.Module):
    """
    one_to_many 논문에서 제시된 resunit 구조
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = self.bn1(x)
        residual = self.relu1(residual)
        residual = self.conv1(residual)
        residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)
        return x + residual


def PixelNormLayer(x):
    """
    Pixelwise feature vector normalization.
    """
    return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)



class ResidualBlock_no_bn(nn.Module):
    """
    one_to_many 논문에서 제시된 resunit 구조
    """
    def __init__(self, channels):
        super(ResidualBlock_no_bn, self).__init__()


        self.conv1 = spectral_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()


        self.conv2 = spectral_norm(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu1(residual)

        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.relu2(residual)

        return x + residual * 0.001


def ResidualBlocks(channels, size):
    bundle = []
    for i in range(size):
        bundle.append(ResidualBlock(channels))
    return nn.Sequential(*bundle)


def ResidualBlocks_no_bn(channels, size):
    bundle = []
    for i in range(size):
        bundle.append(ResidualBlock_no_bn(channels))
    return nn.Sequential(*bundle)