import numpy as np
import torch.nn as nn
from Models.common import ConvBlock_Discriminator
from torchsummary import summary
import torch

class Discriminator(nn.Module):
    def __init__(self, imagesize, initial_channel = 64 ):
        super(Discriminator, self).__init__()
        in_channel = imagesize[0]
#        print("in_channel : {}".format(in_channel))
        ConvBlock_list = []
        ConvBlock_list.append(nn.Conv2d(in_channels=in_channel,out_channels=initial_channel,kernel_size=3,padding=1))
        ConvBlock_list.append(nn.LeakyReLU(negative_slope=0.2,inplace=True))
#        print("initial channel : {}".format(initial_channel))
        ConvBlock_list.append(ConvBlock_Discriminator(in_channel=initial_channel,stride=2))

        channel = initial_channel

        for i in range(3):
            #print("first channel : {}".format(channel))
            ConvBlock_list.append(ConvBlock_Discriminator(in_channel=channel, stride=1))
            channel = channel * 2
            #print("second channel : {}".format(channel))
            if i ==2:
                break;
            ConvBlock_list.append(ConvBlock_Discriminator(in_channel = channel, stride=2))

        ConvBlock_list.append(nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=3, stride=2,padding=1))
        ConvBlock_list.append(nn.BatchNorm2d(channel))
        ConvBlock_list.append(nn.LeakyReLU(negative_slope=0.2))
        ConvBlock_list.append(nn.Flatten())

        self.ConvBlock_module = nn.Sequential(*ConvBlock_list)

        flattensize = int((imagesize[1]//16+1)*(imagesize[2]//16+1)*channel)
        #print(flattensize)
        #print(round(imagesize[0]/16.0))
        self.Dense1 = nn.Linear(in_features=flattensize,out_features=1024)
        self.LeakyReLU2 = nn.LeakyReLU(negative_slope=0.2,inplace=True)
        self.Dense2 = nn.Linear(in_features=1024,out_features=1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = 2.0 * x - 1.0
        #print(torch._shape_as_tensor(x))
        out = self.ConvBlock_module(x)
        #print(torch._shape_as_tensor(out))

        out = self.Dense1(out)
        out = self.LeakyReLU2(out)
        out = self.Dense2(out)
        out = self.Sigmoid(out)

        return out


if __name__ =="__main__":
    test = Discriminator((3,296,296))
    keys = test.state_dict().keys()
    for key in keys:
        if key.find('weight') != -1:
            print("name of layer : {}".format(key))
            print(torch._shape_as_tensor((test.state_dict())[key]))

    summary(test,(3,296,296),device='cpu')
