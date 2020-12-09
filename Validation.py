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

from Models.Generator_128 import Generator

parser = argparse.ArgumentParser(description="SRFeat Training Module")
parser.add_argument('--pre_trained', type=int, default=0, help="path of pretrained models")

BATCH_SIZE = 9
CROP_SIZE = 296
UPSCALE_FACTOR = 4
DIRPATH_TRAIN = "COCO/train2017"
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
    valid_dataset = Dataset_gen.Pretrain_Dataset_Train(dirpath=DIRPATH_TRAIN, crop_size=CROP_SIZE,
                                                       upscale_factor=UPSCALE_FACTOR)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=0,
                                  pin_memory=True)

    generator = Generator()

    validation_PSNR =0
    PRETRAINED_MODELPATH = os.path.join(DIRPATH_PRETRAIN, "generator_{}th_model.pth".format(PRETRAINED_EPOCH - 1))
    generator = utils.load_model(generator, filepath = PRETRAINED_MODELPATH)
    generator = nn.DataParallel(generator)
    generator = generator.to(device)

    accum_psnr = 0
    generator.eval()
    for lr_image, hr_image in tqdm.tqdm(valid_dataloader, bar_format="{l_bar}{bar:40}{r_bar}"):
        # print("---batch {}---".format(i))
        target_list = np.array(hr_image)

        input_list = np.array(lr_image)

        lr_image, hr_image = lr_image.to(device), hr_image.to(device)
        # print(type(input))

        # generate fake hr images
        fake_hr = generator(lr_image)

        pretrain_loss = utils.get_psnr(fake_hr, hr_image)

        temp_psnr = utils.get_psnr(fake_hr, hr_image)
        # accum_psnr += 10 * torch.log10(1 / pretrain_loss)
        accum_psnr += temp_psnr

    validation_PSNR = accum_psnr/len(valid_dataloader)


    print("average PSNR : {}".format(validation_PSNR))

    # if (epoch +1) %10 ==0:
    np.save("Result_image/PSNR_{}".format(PRETRAINED_EPOCH),validation_PSNR)

    #   Train_Gen_loss[epoch] = Gen_loss_total / len(train_dataloader)
    #   Train_Dis_loss[epoch] = Dis_loss_total / len(train_dataloader)
    #   PSNR_train[epoch] = total_PSNR_train / len(train_dataloader)
    #   print("train PSNR : {}".format(total_PSNR_train / len(train_dataloader)))
