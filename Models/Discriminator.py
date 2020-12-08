import numpy as np
import torch.nn as nn
from Models.common import ConvBlock_Discriminator

class Discriminator(nn.Module):
    def __init__(self, imagesize, initial_channel = 64 ):
        super(Discriminator, self).__init__()
        ConvBlock_list = []
        ConvBlock_list.append(nn.Conv2d(in_channels=3,out_channels=initial_channel,kernel_size=3,padding=1))
        ConvBlock_list.append(nn.LeakyReLU(negative_slope=0.2,inplace=True))

        ConvBlock_list.append(ConvBlock_Discriminator(in_channel=initial_channel, stride=2))

        channel = initial_channel*2
        for i in range(3):
            ConvBlock_list.append(ConvBlock_Discriminator(in_channel=channel, stride=1))
            ConvBlock_list.append(ConvBlock_Discriminator(in_channel = channel, stride=2))
            channel = channel*2

        self.ConvBlock_module = nn.Sequential(*ConvBlock_list)

        flattensize = int((imagesize[0]//16)*(imagesize[1]//16)*(channel//2))

        self.Dense1 = nn.Linear(in_features=flattensize,out_features=1024)
        self.LeakyReLU2 = nn.LeakyReLU(negative_slope=0.2,inplace=True)
        self.Dense2 = nn.Linear(in_features=1024,out_features=1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = 2.0 * x - 1.0
        out = self.ConvBlock_module(x)
        out = self.Dense1(out)
        out = self.LeakyReLU2(out)
        out = self.Dense2(out)
        out = self.Sigmoid(out)

        return out


