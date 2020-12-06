import torch.nn as nn
import numpy as np
import torch
from torchsummary import summary #pip install torchsummary

#BN 을 빼고서도 실험해보자.
class Residual_block(nn.Module):
    def __init__(self, channel = 128):
        super(Residual_block, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=3,padding=1)
        self.Conv2 = nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=3,padding=1)
        self.BN1 = nn.BatchNorm2d(64)
        self.BN2 = nn.BatchNorm2d(64)
        self.LeakyReLU1 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self,x):
        out = self.Conv1(x)
        out = self.BN1(out)
        out = self.LeakyReLU1(out)

        out = self.Conv2(out)
        out = self.BN2(out)
        out = out+x
        return out

class Pixelsuffler_block(nn.Module):
    def __init__(self,in_channels):
        super(Pixelsuffler_block, self).__init__()
        self.Conv = nn.Conv2d(in_channels=in_channels,out_channels=in_channels*4,kernel_size=3,padding=1)
        self.PixelSuffer = nn.PixelShuffle(2)

    def forward(self,x):
        out = self.Conv(x)
        out = self.PixelSuffer(out)
        return out


class ConvBlock_Discriminator(nn.Module):
    def __init__(self,in_channel, stride=1):
        super(ConvBlock_Discriminator, self).__init__()
        
        if stride == 1:
            out_channel = in_channel*2
        else:
            out_channel = in_channel

        self.Conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1)
        self.BN = nn.BatchNorm2d(out_channel)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)

    def forward(self,x):
        out = self.Conv(x)
        out = self.BN(out)
        out = self.LeakyReLU(out)

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.LeakyReLU1 = nn.LeakyReLU(negative_slope=0.2)

        ConvBlock_list = []
        ConvBlock_list.append(ConvBlock_Discriminator(64,channel_increase=True))
        ConvBlock_list.append(ConvBlock_Discriminator(128,channel_increase=False))
        ConvBlock_list.append(ConvBlock_Discriminator(128,channel_increase=True))
        ConvBlock_list.append(ConvBlock_Discriminator(256,channel_increase=False))
        ConvBlock_list.append(ConvBlock_Discriminator(256,channel_increase=True))
        ConvBlock_list.append(ConvBlock_Discriminator(512,channel_increase=False))

        ConvBlock_list.append(nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=2,padding=1))
        ConvBlock_list.append(nn.BatchNorm2d(512))
        ConvBlock_list.append(nn.LeakyReLU(0.2))
        """
        for i in range(3):
            ConvBlock_list.append(ConvBlock_Discriminator(ConvBlock_input,channel_increase=False))
            ConvBlock_list.append(ConvBlock_Discriminator(ConvBlock_input, channel_increase=True))
            ConvBlock_input = ConvBlock_input*2 
        """
        self.ConvBlock_module = nn.Sequential(*ConvBlock_list)

        self.Avgpooling = nn.AdaptiveAvgPool2d(1)
        self.Dense1 = nn.Linear(in_features=512,out_features=1024)
        self.LeakyReLU2 = nn.LeakyReLU(0.2)
        self.Dense2 = nn.Linear(in_features=1024,out_features=1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self,x):

        out_block1 = self.Conv1(x)
        out_block1 = self.LeakyReLU1(out_block1)

        #start block like VGG
        out_block2 = self.ConvBlock_module(out_block1)

        input_block3 = self.Avgpooling(out_block2)
        input_block3 = torch.squeeze(input_block3)
        out_block3 = self.Dense1(input_block3)
        out_block3 = self.LeakyReLU2(out_block3)
        out_block3 = self.Dense2(out_block3)
        out_block3 = self.Sigmoid(out_block3)

        return out_block3



if __name__ == "__main__":
    TestGenerator = Generator().to('cuda:0')
    summary(TestGenerator,(3,96,96))
    TestDiscriminator = Discriminator().to('cuda:0')
    summary(TestDiscriminator,(3,384,384))




