import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import Dataset_gen
import tqdm

import Model
from Models import Truncated_vgg
import argparse
import os
import utils

from Models.Generator_128 import Generator

parser = argparse.ArgumentParser(description="SRGAN Training Module")
parser.add_argument('--pre_trained', type=str, default=None, help="path of pretrained models")
parser.add_argument('--num_epochs', type=int, default=20, help="train epoch")
parser.add_argument('--pre_resulted', type=str, default=None, help="data of previous step")

BATCH_SIZE = 9
CROP_SIZE = 296
UPSCALE_FACTOR = 4
DIRPATH_PRETRAIN = "COCO/train2017"

TOTAL_EPOCHS = 20
grad_clip = None
lr = 1e-4

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu:0")

if __name__ == "__main__":

    opt = parser.parse_args()

    PRETRAINED_PATH = opt.pre_trained
    TOTAL_EPOCHS = opt.num_epochs
    PRE_RESULT_DIR = opt.pre_resulted

    loss_array_Train = np.zeros(TOTAL_EPOCHS)
    PSNR_array_Train = np.zeros(TOTAL_EPOCHS)
    PSNR_array_Vaild = np.zeros(TOTAL_EPOCHS)

    DIRPATH_PRETRAIN = DIRPATH_PRETRAIN
    train_dataset = Dataset_gen.Pretrain_Dataset_Train(dirpath=DIRPATH_PRETRAIN, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)

    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


    generator = Generator()
    gen_optimizer = optim.Adam(generator.parameters(), lr=lr)  # lr = 1e-4
    scheduler = optim.lr_scheduler.CosineAnnealingLR(gen_optimizer,T_max = TOTAL_EPOCHS//2, eta_min=1e-6)
    mseloss = nn.MSELoss()

    PSNR_train = np.zeros(TOTAL_EPOCHS)
    Train_Gen_loss = np.zeros(TOTAL_EPOCHS)
    datasize = len(train_dataloader)

    start_epoch = 0
    
 #   generator = generator.to(device)
#    generator = nn.DataParallel(generator)
  #  generator = generator.to('cuda')
    generator = generator.cuda()

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        # prepare training
        generator.train()

        total_MSE_train = 0
        accum_psnr = 0

        print("----epoch {}/{}----".format(epoch + 1, TOTAL_EPOCHS))
        print("----training step----")
        for lr_image,hr_image in tqdm.tqdm(train_dataloader, bar_format="{l_bar}{bar:40}{r_bar}"):
            # print("---batch {}---".format(i))
            target_list = np.array(hr_image)
            
            input_list = np.array(lr_image)

            lr_image, hr_image = lr_image.to(device), hr_image.to(device)
           # print(type(input))
           # print(type(generator))
            gen_optimizer.zero_grad()

            # generate fake hr images
            fake_hr = generator(lr_image)

            pretrain_loss = mseloss(fake_hr,hr_image)

            total_MSE_train += pretrain_loss
            accum_psnr += 10 * torch.log10(1 / pretrain_loss)

            pretrain_loss.backward()
            gen_optimizer.step()

        if epoch<TOTAL_EPOCHS//2:
            scheduler.step()

        PSNR_train[epoch] = accum_psnr/datasize
        Train_Gen_loss[epoch] = total_MSE_train/datasize

        print("average PSNR : {} | MSE : {}".format(PSNR_train[epoch],Train_Gen_loss[epoch]))

        if (epoch +1) %10 ==0:
            torch.save(generator.state_dict(), "Trained_model/Generator/generator_{}th_model.pth".format(epoch))
    #   Train_Gen_loss[epoch] = Gen_loss_total / len(train_dataloader)
    #   Train_Dis_loss[epoch] = Dis_loss_total / len(train_dataloader)
    #   PSNR_train[epoch] = total_PSNR_train / len(train_dataloader)
    #   print("train PSNR : {}".format(total_PSNR_train / len(train_dataloader)))
