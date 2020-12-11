import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import Dataset_gen
import tqdm

import argparse
import os
import utils
import glob

from Models.Generator_128 import Generator
from piq import psnr

parser = argparse.ArgumentParser(description="SRFeat Validation Module")
parser.add_argument('--pre_trained', type=int, default=0, help="path of pretrained models")

BATCH_SIZE = 9
CROP_SIZE = 296
UPSCALE_FACTOR = 4
DIRPATH_Validation = "Dataset/valid"
DIRPATH_PRETRAIN = "Trained_model/Generator"

TOTAL_EPOCHS = 20
grad_clip = None
lr = 1e-4

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu:0")

if __name__ == "__main__":

    opt = parser.parse_args()

    PRETRAINED_EPOCH = opt.pre_trained

    #    utils.remove_small_images(DIRPATH_PRETRAIN,minimum=296)
    datasetlist = glob.glob(os.path.join(DIRPATH_Validation,"*"))

    valid_dataloaderlist = []
    for datasetname in datasetlist:
        temp_valid_dataset = Dataset_gen.Dataset_Validation(dirpath=datasetname, crop_size=CROP_SIZE,
                                                       upscale_factor=UPSCALE_FACTOR)
        temp_valid_dataloader = DataLoader(dataset=temp_valid_dataset, batch_size=4, shuffle=False, num_workers=2,
                                  pin_memory=True)
        valid_dataloaderlist.append(temp_valid_dataloader)

    generator = Generator()

    validation_PSNR =0
    PRETRAINED_MODELPATH = os.path.join(DIRPATH_PRETRAIN, "generator_4th_model.pth")
    generator = utils.load_model(generator, filepath = PRETRAINED_MODELPATH,device =device)
    #generator = nn.DataParallel(generator
    generator = generator.to(device)

    accum_psnr = 0
    generator.eval()
    for i,valid_dataloader in enumerate(valid_dataloaderlist):
        print("validate about dataset {}".format(datasetlist[i]))
        for lr_image, hr_image in tqdm.tqdm(valid_dataloader, bar_format="{l_bar}{bar:40}{r_bar}"):
            # print("---batch {}---".format(i))
            lr_image, hr_image = lr_image.to(device), hr_image.to(device)

            # generate fake hr images
            fake_hr = generator(lr_image)
            print("value range : {} to {}".format(torch.min(fake_hr),torch.max(fake_hr)))
            accum_psnr += psnr(fake_hr, hr_image)
            # accum_psnr += 10 * torch.log10(1 / pretrain_loss)
        validation_PSNR = accum_psnr/len(valid_dataloader)
        print("average PSNR about dataset {}: {}".format(datasetlist[i],validation_PSNR))

    #   Train_Gen_loss[epoch] = Gen_loss_total / len(train_dataloader)
    #   Train_Dis_loss[epoch] = Dis_loss_total / len(train_dataloader)
    #   PSNR_train[epoch] = total_PSNR_train / len(train_dataloader)
    #   print("train PSNR : {}".format(total_PSNR_train / len(train_dataloader)))
