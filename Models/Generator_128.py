import numpy as np
import torch.nn as nn

from Models.common import Residual_block, Pixelsuffler_block

class Generator(nn.Module):
    def __init__(self, channelsize = 128, scale_factor = 4, resblock_num = 16):
        super(Generator, self).__init__()
        self.pixelsuffle_layer_num = int(np.log2(scale_factor))

        self.Conv1 = nn.Conv2d(in_channels=3,out_channels=channelsize,kernel_size=9, padding=4)

        self.set_resblock = []
        for i in range(resblock_num):
            resblock = Residual_block(channel=channelsize)
            self.set_resblock.append(resblock)

        self.set_longrange_connection = []
        for i in range(resblock_num-1):
            skipconnection = nn.Conv2d(in_channels=channelsize,out_channels=channelsize,kernel_size=1)
            self.set_longrange_connection.append(skipconnection)

        Upsampling_layers = []
        for _ in range(self.pixelsuffle_layer_num):
            Upsampling_layers.append(Pixelsuffler_block(in_channels=channelsize)) #2배씩 upscaling
        self.upsmapling_module = nn.Sequential(*Upsampling_layers)

        self.lastconv = nn.Conv2d(in_channels=channelsize, out_channels=3, kernel_size=3, padding=1)


    def forward(self,x):
        #first conv
        first_feature = self.Conv1(x)

        set_resout = []
        pre_out = first_feature
        for resblock in self.resblock_set:
            pre_out = resblock(pre_out)
            set_resout.append(pre_out)

        extracted_feature= set_resout[-1]
        for i,resout in enumerate(set_resout[:-1]):
            skipconnection = self.set_longrange_connection[i]
            extracted_feature += skipconnection(resout)

        recon_out = self.upsmapling_module(extracted_feature)
        out = self.lastconv(recon_out)

        #passing 5 residual blocks
        return out

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                k = 1/np.sqrt(fan_in)
                nn.init.uniform_(m.weight, a=-k,b=k)
                nn.init.uniform_(m.bias, a=-k,b=k)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
