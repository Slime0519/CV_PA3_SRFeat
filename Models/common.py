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
        self.BN1 = nn.BatchNorm2d(channel)
        self.BN2 = nn.BatchNorm2d(channel)
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
        
        if stride> 1:
            out_channel = in_channel
        else:
            out_channel = in_channel*2

        self.Conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1)
        self.BN = nn.BatchNorm2d(out_channel)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)

    def forward(self,x):
        out = self.Conv(x)
        out = self.BN(out)
        out = self.LeakyReLU(out)

        return out

"""
class WrapperModel(nn.Module):
	def __init__(self):
		super(WrapperModel, self).__init__()
		self.module = Model() # that I actually define.
	def forward(self, x):
		return self.module(x)
"""

#if __name__ == "__main__":
   #  TestGenerator = Generator().to('cuda:0')
   # summary(TestGenerator,(3,96,96))
   # TestDiscriminator = Discriminator().to('cuda:0')
   # summary(TestDiscriminator,(3,384,384))
#



