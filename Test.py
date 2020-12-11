import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
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
DIRPATH_Test = "Dataset/valid"
DIRPATH_PRETRAIN = "Trained_model/Generator"
DIRPATH_TESTIMAGE = "Test_result"
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
    datasetpath_list = glob.glob(os.path.join(DIRPATH_Test,"*"))

    test_dataloaderlist = []
    datasetname_list = []
    for datasetpath in datasetpath_list:
        temp_test_dataset = Dataset_gen.Dataset_Test(dirpath=datasetpath)
        temp_test_dataloader = DataLoader(dataset=temp_test_dataset, batch_size=1, shuffle=False, num_workers=0)
        test_dataloaderlist.append(temp_test_dataloader)

        datasetname = datasetpath.split('/')[-1]
        datasetname_list.append(datasetname)


    generator = Generator()


    PRETRAINED_MODELPATH = os.path.join(DIRPATH_PRETRAIN, "generator_4th_model.pth")
    generator = utils.load_model(generator, filepath = PRETRAINED_MODELPATH,device =device)
    #generator = nn.DataParallel(generator)
    generator = generator.to(device)

    generator.eval()

    for i,test_dataloader in enumerate(test_dataloaderlist):
        print("validate about dataset {}".format(datasetname_list[i]))
        imagenum = 0
        for lr_image in tqdm.tqdm(test_dataloader, bar_format="{l_bar}{bar:40}{r_bar}"):
            # print("---batch {}---".format(i))
            lr_image = lr_image.to(device)

            # generate fake hr images
            fake_hr = generator(lr_image)
            temp_fake = torch.clamp(fake_hr, min=0, max=1)

            saveimage = temp_fake[0]
            save_image(saveimage,os.path.join(DIRPATH_TESTIMAGE,datasetname_list[i],"image{}.png".format(imagenum)))
            imagenum +=1

       # print("average PSNR about dataset {}: {}".format(datasetname_list[i],validation_PSNR))
