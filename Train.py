import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import Dataset_gen
import Model
from Models import Truncated_vgg
import argparse
import os
import utils

from Models.Generator_128 import Generator
from Models.Discriminator import Discriminator
from Models.Truncated_vgg import truncated_vgg

parser = argparse.ArgumentParser(description="SRFeat Training Module")
parser.add_argument('--pre_trained', type = str, default=None, help = "path of pretrained models")
parser.add_argument('--num_epochs', type = int, default=100, help="train epoch")
parser.add_argument('--pre_resulted', type = str, default=None,  help = "data of previous step")

BATCH_SIZE = 16
CROP_SIZE = 96
UPSCALE_FACTOR = 4
DIRPATH_TRAIN = "Dataset/Train"
DIRPATH_VAILD = "Dataset/Vaild"
TOTAL_EPOCH = 100
grad_clip = None
lr = 1e-4

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu:0")

if __name__ == "__main__":

    opt = parser.parse_args()

    PRETRAINED_PATH = opt.pre_trained
    TOTAL_EPOCH = opt.num_epochs
    PRE_RESULT_DIR = opt.pre_resulted

    train_dataset = Dataset_gen.Dataset_Train(dirpath=DIRPATH_TRAIN, crop_size=96, upscale_factor=UPSCALE_FACTOR)
    vaild_dataset = Dataset_gen.Dataset_Vaild(dirpath=DIRPATH_VAILD, upscale_factor=UPSCALE_FACTOR)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    vaild_dataloader = DataLoader(dataset=vaild_dataset, batch_size=1)

    generator = Generator()
    image_discriminator = Discriminator(imagesize = (294,294))
    feat_discriminator = Discriminator(imagesize = (24,24))

    gen_optimizer = optim.Adam(generator.parameters(),lr= lr) #lr = 1e-4
    imgdis_optimizer = optim.Adam(image_discriminator.parameters(),lr = lr)
    featdis_optimizer = optim.Adam(feat_discriminator.parameters(),lr=lr)
    truncat_vgg = truncated_vgg()  # vgg(5,4) loss

    mseloss = nn.MSELoss()
    adversal_criterion = nn.BCEWithLogitsLoss()

    PSNR_eval = np.zeros(TOTAL_EPOCH)
    PSNR_train = np.zeros(TOTAL_EPOCH)
    Train_Gen_loss = np.zeros(TOTAL_EPOCH)
    Train_Dis_loss = np.zeros(TOTAL_EPOCH)
    train_len = len(train_dataloader)

    start_epoch = 0
    """
    if PRETRAINED_PATH is not None:
        _, generatorpath = utils.load_module(os.path.join(PRETRAINED_PATH, "Generator"))
        start_epoch, dis_modelpath = utils.load_module(os.path.join(PRETRAINED_PATH, "Discriminator"))
        print(dis_modelpath)
        print("load module : saved on {} epoch".format(start_epoch))
        generator.load_state_dict(torch.load(generatorpath))
        Dis_Model.load_state_dict(torch.load(dis_modelpath))

    if PRE_RESULT_DIR is not None:
        PSNR_eval = np.load("result_data/PSNR_eval.npy")
        PSNR_Train = np.load("result_data/PSNR_train.npy")
        Train_Dis_loss = np.load("result_data/Train_Dis_loss.npy")
        Train_Gen_loss = np.load("result_data/Train_Gen_loss.npy")
    """
    generator = generator.to(device)
    image_discriminator = image_discriminator.to(device)
    truncat_vgg = truncat_vgg.to(device)


    for epoch in range(start_epoch,TOTAL_EPOCH):
        # prepare training
        generator.train()
        image_discriminator.train()
        feat_discriminator.train()

        Gen_loss_total = 0
        imgdis_loss_total = 0
        featdis_loss_total = 0

        total_PSNR_train = 0
        print("----epoch {}/{}----".format(epoch+1, TOTAL_EPOCH))
        print("----training step----")
        for i, (input, target) in enumerate(train_dataloader):
           # print("---batch {}---".format(i))
            target_list = np.array(target)
            input_list = np.array(input)
            """
            for i, target_image in enumerate(target_list):
                print("target {} : {}".format(i,np.array(target_image).shape))
            for i,input_image in enumerate(input):
                print("input {}: {}".format(i,np.array(input_image).shape))
            """
            input, target = input.to(device), target.to(device)

            imgdis_optimizer.zero_grad()
            gen_optimizer.zero_grad()
            featdis_optimizer.zero_grad()

            #generate fake hr images
            fake_hr = generator(input)
           # generate vgg pathces
            fake_vgg_patch = truncat_vgg(fake_hr)/12.75
            real_vgg_patch = truncat_vgg(target)/12.75

           # train image Discriminator
            img_fake_crimed = image_discriminator(fake_hr)
            img_real_crimed = image_discriminator(target)

            fake_score = adversal_criterion(img_fake_crimed,torch.zeros_like(img_fake_crimed))
            target_score = adversal_criterion(img_real_crimed,torch.ones_like(img_real_crimed))

            img_adversarial_loss = fake_score+target_score

            img_adversarial_loss.backward()
            imgdis_optimizer.step()

            #train feature Discriminator
            feat_fake_crimed = feat_discriminator(fake_vgg_patch)
            feat_real_crimed = feat_discriminator(real_vgg_patch)

            feat_fake_score = adversal_criterion(feat_fake_crimed,torch.zeros_like(feat_fake_crimed))
            feat_real_score = adversal_criterion(feat_real_crimed,torch.ones_like(feat_real_crimed))

            feat_adversarial_loss = feat_fake_score+feat_real_score
            feat_adversarial_loss.backward()
            featdis_optimizer.step()

            # train Generator
            imgdis_optimizer.requires_grad_(False)
            featdis_optimizer.requires_grad_(False)

            gen_image_crimed = image_discriminator(fake_hr)
            gen_feat_crimed = feat_discriminator(fake_hr)

            gen_img_score = adversal_criterion(gen_image_crimed,torch.ones_like(gen_image_crimed))
            gen_feat_score = adversal_criterion(gen_feat_score,torch.ones_like(gen_feat_score))
            perceptual_loss = mseloss(fake_vgg_patch,real_vgg_patch)
            #common_loss = mseloss(fake_hr,target)

            gen_total_loss = perceptual_loss + 1e-3*(gen_img_score+gen_feat_score)

            gen_total_loss.backward()
            gen_optimizer.step()
            print("epoch {} training step : {}/{}".format(epoch+1, i + 1, train_len))

     #   Train_Gen_loss[epoch] = Gen_loss_total / len(train_dataloader)
     #   Train_Dis_loss[epoch] = Dis_loss_total / len(train_dataloader)
     #   PSNR_train[epoch] = total_PSNR_train / len(train_dataloader)
     #   print("train PSNR : {}".format(total_PSNR_train / len(train_dataloader)))
    """
        generator.eval()
        Dis_Model.eval()
        total_PSNR_eval = 0
        print("----evaluation step----")
        with torch.no_grad():
            # val_bar = tqdm(vaild_dataloader)
            for input, target in vaild_dataloader:
                input = input.to(device)
                fakeimage = generator(input)
                fakeimage = np.array(fakeimage.cpu().detach())
                fakeimage = fakeimage.squeeze()
                print(fakeimage.shape)

                target = np.array(target.detach())
                batch_MSE = np.mean((fakeimage - target) ** 2)
                PSNR_temp = 10 * np.log10(1 / batch_MSE)
                total_PSNR_eval += PSNR_temp

            PSNR_eval[epoch] = total_PSNR_eval / len(vaild_dataloader)
            print("evaluation PSNR : {}".format(total_PSNR_eval / len(vaild_dataloader)))
            
    
        np.save("result_data/Train_Gen_loss.npy",Train_Gen_loss)
        np.save("result_data/Train_Dis_loss.npy", Train_Dis_loss)
        np.save("result_data/PSNR_train.npy",PSNR_train)
        np.save("result_data/PSNR_eval.npy",PSNR_eval)
    
        torch.save(generator.state_dict(), "Trained_model/Generator/generator_{}th_model.pth".format(epoch))
        torch.save(feat_discriminator.state_dict(), "Trained_model/Discriminator/discriminator_{}th_model.pth".format(epoch))
        torch.save(feat_discriminator.state_dict(), "Trained_model/Discriminator/discriminator_{}th_model.pth".format(epoch))
    """
