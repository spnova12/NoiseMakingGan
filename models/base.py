import math
from models.subNets import *
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.utils.spectral_norm as spectral_norm

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #############
        # Noise
        #############
        self.z_conv1 = spectral_norm(nn.Conv2d(1, 64,  kernel_size=3, stride=1, padding=1))
        self.z_bn1 = nn.BatchNorm2d(64)
        self.z_lrelu1 = nn.ReLU()

        self.z_rb1 = ResidualBlocks_no_bn(64, 5)


        #############
        # camera info
        #############
        self.caminfo_conv1 = spectral_norm(nn.Conv2d(3, 128,  kernel_size=3, stride=1, padding=1))
        self.caminfo_bn1 = nn.BatchNorm2d(128)
        self.caminfo_lrelu1 = nn.ReLU()

        self.caminfo_rb1 = ResidualBlocks_no_bn(128, 5)


        #############
        # camera info
        #############
        self.img_conv1 = spectral_norm(nn.Conv2d(3, 128,  kernel_size=3, stride=1, padding=1))
        self.img_bn1 = nn.BatchNorm2d(128)
        self.img_lrelu1 = nn.ReLU()

        self.img_rb1 = ResidualBlocks_no_bn(128, 5)


        #############
        # Final layers
        #############
        self.compress1 = spectral_norm(nn.Conv2d(64 + 128 * 2, 256, kernel_size=1, stride=1, padding=0))
        self.bn1 = nn.BatchNorm2d(256)
        self.lrelu1 = nn.ReLU()

        self.rb1 = ResidualBlocks_no_bn(256, 3)


        self.compress2 = spectral_norm(nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0))
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.ReLU()

        self.rb2 = ResidualBlocks_no_bn(128, 2)


        self.compress3 = spectral_norm(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0))
        self.bn3 = nn.BatchNorm2d(64)
        self.lrelu3 = nn.ReLU()

        self.rb3 = ResidualBlocks_no_bn(64, 5)


        self.last = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)


    def forward(self, z, camera_info_tensor, input_img):
        out_z = self.z_conv1(z)
        out_z = self.z_bn1(out_z)
        out_z = self.z_lrelu1(out_z)
        out_z = self.z_rb1(out_z)

        out_caminfo = self.caminfo_conv1(camera_info_tensor)
        out_caminfo = self.caminfo_bn1(out_caminfo)
        out_caminfo = self.caminfo_lrelu1(out_caminfo)
        out_caminfo = self.caminfo_rb1(out_caminfo)

        out_img = self.img_conv1(input_img)
        out_img = self.img_bn1(out_img)
        out_img = self.img_lrelu1(out_img)
        out_img = self.img_rb1(out_img)

        out = torch.cat((out_z, out_caminfo, out_img), 1)

        out = self.compress1(out)
        out = self.bn1(out)
        out = self.lrelu1(out)
        out = self.rb1(out)

        out = self.compress2(out)
        out = self.bn2(out)
        out = self.lrelu2(out)
        out = self.rb2(out)

        out = self.compress3(out)
        out = self.bn3(out)
        out = self.lrelu3(out)
        out = self.rb3(out)

        out = self.last(out)

        return out


#
#
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         input_channel = 12
#
#         self.layer1 = torch.nn.Conv2d(input_channel, 64, kernel_size=4, stride=1, padding=1)
#         self.layer2 = torch.nn.LeakyReLU(0.1, inplace=True)
#         self.layer3 = torch.nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
#         self.layer4 = torch.nn.LeakyReLU(0.1, inplace=True)
#         self.layer5 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1)
#         self.layer6 = torch.nn.LeakyReLU(0.1, inplace=True)
#         self.layer7 = torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
#         self.layer8 = torch.nn.LeakyReLU(0.1, inplace=True)
#         self.layer9 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1)
#         self.layer10 = torch.nn.LeakyReLU(0.1, inplace=True)
#         self.layer11 = torch.nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
#         self.layer12 = torch.nn.LeakyReLU(0.1, inplace=True)
#         self.layer13 = torch.nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)
#         self.layer14 = torch.nn.LeakyReLU(0.1, inplace=True)
#         self.layer15 = torch.nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
#         self.layer16 = torch.nn.LeakyReLU(0.1, inplace=True)
#
#
#
#         self.fc = nn.Linear(2048, 1)
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features
#
#     def conv(self, c_in, c_out, k_size, stride=2, pad=1, bn=True):
#         """
#         Custom convolutional layer for simplicity.
#         bn 을 편하게 사용하기 위해 만든 함수
#         """
#         layers = []
#         layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
#         if bn:
#             layers.append(nn.BatchNorm2d(c_out))
#         return nn.Sequential(*layers)
#
#     def forward(self, camera_info_tensor, input_img, noise1, noise2):
#         x = torch.cat((camera_info_tensor, input_img, noise1, noise2), 1)
#
#         # out = PixelNormLayer(self.layer1(x))
#         # out = self.layer2(out)
#         # out = PixelNormLayer(self.layer3(out))
#         # out = self.layer4(out)
#         # out = PixelNormLayer(self.layer5(out))
#         # out = self.layer6(out)
#         # out = PixelNormLayer(self.layer7(out))
#         # out = self.layer8(out)
#         # out = PixelNormLayer(self.layer9(out))
#         # out = self.layer10(out)
#         # out = PixelNormLayer(self.layer11(out))
#         # out = self.layer12(out)
#         # out = PixelNormLayer(self.layer13(out))
#         # out = self.layer14(out)
#         # out = self.relu2 = nn.LeakyReLU(0.2)(self.layer15(out))
#         # out = self.layer16(out)
#
#         out = (self.layer1(x))
#         out = self.layer2(out)
#         out = (self.layer3(out))
#         out = self.layer4(out)
#         out = (self.layer5(out))
#         out = self.layer6(out)
#         out = (self.layer7(out))
#         out = self.layer8(out)
#         out = (self.layer9(out))
#         out = self.layer10(out)
#         out = (self.layer11(out))
#         out = self.layer12(out)
#         out = (self.layer13(out))
#         out = self.layer14(out)
#         out = (self.layer15(out))
#         out = self.layer16(out)
#
#         # logistic regression and sigmoid
#         out = out.view(-1, self.num_flat_features(out))
#
#         out = self.fc(out)
#         out = torch.sigmoid(out)
#         return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        input_channel = 12

        model = [
            spectral_norm(torch.nn.Conv2d(input_channel, 64, kernel_size=4, stride=1, padding=1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(torch.nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(torch.nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(torch.nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(torch.nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(torch.nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(torch.nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
        ]
        self.model = torch.nn.Sequential(*model)

        self.fc = nn.Linear(2048, 1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def conv(self, c_in, c_out, k_size, stride=2, pad=1, bn=True):
        """
        Custom convolutional layer for simplicity.
        bn 을 편하게 사용하기 위해 만든 함수
        """
        layers = []
        layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
        if bn:
            layers.append(nn.BatchNorm2d(c_out))
        return nn.Sequential(*layers)

    def forward(self, camera_info_tensor, input_img, noise1, noise2):
        x = torch.cat((camera_info_tensor, input_img, noise1, noise2), 1)

        out = self.model(x)

        # logistic regression and sigmoid
        out = out.view(-1, self.num_flat_features(out))
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out
