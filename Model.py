import torch.nn as nn
import numpy as np
import torch
from torchsummary import summary #pip install torchsummary

class Residual_block_Generator(nn.Module):
    def __init__(self):
        super(Residual_block_Generator, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        self.Conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        self.BN1 = nn.BatchNorm2d(64)
        self.BN2 = nn.BatchNorm2d(64)
        self.PReLU1 = nn.PReLU()

    def forward(self,x):
        out = self.Conv1(x)
        out = self.BN1(out)
        out = self.PReLU1(out)

        out = self.Conv2(out)
        out = self.BN2(out)
        out = out+x
        return out

class Pixelsuffer_block_Generator(nn.Module):
    def __init__(self,input_channels, upscaling_factor=2):
        super(Pixelsuffer_block_Generator, self).__init__()
        self.Conv = nn.Conv2d(in_channels=input_channels,out_channels=input_channels*(upscaling_factor**2),kernel_size=3,padding=1,bias=True)
        self.PixelSuffer = nn.PixelShuffle(upscaling_factor)
        self.PReLU = nn.PReLU()

    def forward(self,x):
        out = self.Conv(x)
        out = self.PixelSuffer(out)
        out = self.PReLU(out)

        return out

class Generator(nn.Module):
    def __init__(self, scale_factor = 4, residual_depth = 16):
        super(Generator, self).__init__()
        self.pixelsuffle_layer_num = int(np.log2(scale_factor))

        self.B = residual_depth
        self.Conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=9, padding=4,bias = True)
        self.PReLU1 = nn.PReLU()

        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,bias=True)
        self.BN1 = nn.BatchNorm2d(64)


        #define residual block
        self.RBlock1 = Residual_block_Generator()
        self.RBlock2 = Residual_block_Generator()
        self.RBlock3 = Residual_block_Generator()
        self.RBlock4 = Residual_block_Generator()
        self.RBlock5 = Residual_block_Generator()

        Pixelsuffle_Modulelist = []
        for _ in range(self.pixelsuffle_layer_num):
            Pixelsuffle_Modulelist.append(Pixelsuffer_block_Generator(64,2)) #2배씩 upscaling
        self.Pixelsuffle_Module = nn.Sequential(*Pixelsuffle_Modulelist)

        self.Conv5 = nn.Conv2d(in_channels=64,out_channels=3, kernel_size=9, padding=4, bias = False)

    def forward(self,x):
        #first conv
        out_block1 = self.Conv1(x)
        out_block1 = self.PReLU1(out_block1)

        #passing 5 residual blocks
        out_block2 = self.RBlock1(out_block1)
        out_block2 = self.RBlock2(out_block2)
        out_block2 = self.RBlock3(out_block2)
        out_block2 = self.RBlock4(out_block2)
        out_block2 = self.RBlock5(out_block2)

        out_block3 = self.Conv3(out_block2)
        out_block3 = self.BN1(out_block3)

        input_block4 = out_block3 + out_block1
        out_block4 = self.Pixelsuffle_Module(input_block4)

        final_output = self.Conv5(out_block4)

      #  return torch.tanh(final_output+1)/2 #이유 찾아보자.
        return final_output

class ConvBlock_Discriminator(nn.Module):
    def __init__(self,input_channel, channel_increase = False):
        super(ConvBlock_Discriminator, self).__init__()
        if channel_increase:
            change_factor = 2
        else:
            change_factor = 1

        output_channel = input_channel*change_factor

        self.Conv = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=change_factor, padding=1)
        self.BN = nn.BatchNorm2d(output_channel)
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




