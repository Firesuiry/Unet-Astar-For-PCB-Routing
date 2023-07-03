import torch
import torch.nn as nn
from torchvision import models


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class ResNetUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = DoubleConv(in_ch, 32)  # n 256 256 32
        self.pool1 = nn.MaxPool2d(2)  # n 128 128 32
        self.conv2 = DoubleConv(32, 64)  # n 128 128 64
        self.pool2 = nn.MaxPool2d(2)  # n 64 64 64
        self.conv3 = DoubleConv(64, 128)  # n 64 64 128
        self.pool3 = nn.MaxPool2d(2)  # n 32 32 128
        self.conv4 = DoubleConv(128, 256)  # n 32 32 256
        self.pool4 = nn.MaxPool2d(2)  # n 16 16 256
        self.conv5 = DoubleConv(256, 512)  # n 16 16 512
        self.pool5 = nn.MaxPool2d(2)  # n 8 8 512
        self.conv6 = DoubleConv(512, 1024)  # n 8 8 1024
        self.pool6 = nn.MaxPool2d(2)  # n 4 4 1024
        self.conv7 = DoubleConv(1024, 1024)  # n 4 4 1024
        self.pool7 = nn.MaxPool2d(2)  # n 2 2 1024
        self.conv8 = DoubleConv(1024, 1024)  # n 2 2 1024
        self.pool8 = nn.MaxPool2d(2)  # n 1 1 1024
        # fully connected layer
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        # up sampling
        self.up9 = nn.ConvTranspose2d(1024, 1024, 2, stride=2)  # n 2 2 1024
        self.conv9 = DoubleConv(1024 + 1024, 1024)  # n 2 2 1024
        self.up10 = nn.ConvTranspose2d(1024, 1024, 2, stride=2)  # n 4 4 1024
        self.conv10 = DoubleConv(1024 + 1024, 1024)  # n 4 4 1024
        self.up11 = nn.ConvTranspose2d(1024, 1024, 2, stride=2)  # n 8 8 1024
        self.conv11 = DoubleConv(1024 + 1024, 1024)  # n 8 8 1024
        self.up12 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  # n 16 16 512
        self.conv12 = DoubleConv(512 + 512, 512)  # n 16 16 512
        self.up13 = nn.ConvTranspose2d(512, 256, 2, stride=2)  # n 32 32 256
        self.conv13 = DoubleConv(256 + 256, 256)  # n 32 32 256
        self.up14 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # n 64 64 128
        self.conv14 = DoubleConv(128 + 128, 128)  # n 64 64 128
        self.up15 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # n 128 128 64
        self.conv15 = DoubleConv(64 + 64, 64)  # n 128 128 64
        self.up16 = nn.ConvTranspose2d(64, 32, 2, stride=2)  # n 256 256 32
        self.conv16 = DoubleConv(32 + 32, 32)  # n 256 256 32
        self.conv17 = nn.Conv2d(32, out_ch, 1)  # n 256 256 1


    def forward(self, input):
        c1 = self.conv1(input)  # n 256 256 32
        p1 = self.pool1(c1)  # n 128 128 32
        c2 = self.conv2(p1)  # n 128 128 64
        p2 = self.pool2(c2)  # n 64 64 64
        c3 = self.conv3(p2)  # n 64 64 128
        p3 = self.pool3(c3)  # n 32 32 128
        c4 = self.conv4(p3)  # n 32 32 256
        p4 = self.pool4(c4)  # n 16 16 256
        c5 = self.conv5(p4)  # n 16 16 512
        p5 = self.pool5(c5)  # n 8 8 512
        c6 = self.conv6(p5)  # n 8 8 1024
        p6 = self.pool6(c6)  # n 4 4 1024
        c7 = self.conv7(p6)  # n 4 4 1024
        p7 = self.pool7(c7)  # n 2 2 1024
        c8 = self.conv8(p7)  # n 2 2 1024
        p8 = self.pool8(c8)  # n 1 1 1024
        # fully connected layer
        p8 = p8.view(p8.size(0), -1)  # n 1024
        p8 = self.fc1(p8)  # n 1024
        p8 = p8.view(p8.size(0), 1024, 1, 1)  # n 1024 1 1
        # up sampling
        up_9 = self.up9(p8)  # n 2 2 1024
        merge9 = torch.cat([up_9, c8], dim=1)  # n 2 2 2048
        c9 = self.conv9(merge9)  # n 2 2 1024
        up_10 = self.up10(c9)  # n 4 4 1024
        merge10 = torch.cat([up_10, c7], dim=1)  # n 4 4 2048
        c10 = self.conv10(merge10)  # n 4 4 1024
        up_11 = self.up11(c10)  # n 8 8 1024
        merge11 = torch.cat([up_11, c6], dim=1)  # n 8 8 2048
        c11 = self.conv11(merge11)  # n 8 8 1024
        up_12 = self.up12(c11)  # n 16 16 512
        merge12 = torch.cat([up_12, c5], dim=1)  # n 16 16 1024
        c12 = self.conv12(merge12)  # n 16 16 512
        up_13 = self.up13(c12)  # n 32 32 256
        merge13 = torch.cat([up_13, c4], dim=1)  # n 32 32 512
        c13 = self.conv13(merge13)  # n 32 32 256
        up_14 = self.up14(c13)  # n 64 64 128
        merge14 = torch.cat([up_14, c3], dim=1)  # n 64 64 256
        c14 = self.conv14(merge14)  # n 64 64 128
        up_15 = self.up15(c14)  # n 128 128 64
        merge15 = torch.cat([up_15, c2], dim=1)  # n 128 128 128
        c15 = self.conv15(merge15)  # n 128 128 64
        up_16 = self.up16(c15)  # n 256 256 32
        merge16 = torch.cat([up_16, c1], dim=1)  # n 256 256 64
        c16 = self.conv16(merge16)  # n 256 256 32
        c17 = self.conv17(c16)  # n 256 256 1
        return c17

