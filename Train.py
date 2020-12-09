import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import Dataset_gen
import argparse
import os
import utils
import tqdm

from Models.Generator_128 import Generator
from Models.Discriminator import Discriminator
from Models.Truncated_vgg import truncated_vgg

parser = argparse.ArgumentParser(description="SRFeat Training Module")
parser.add_argument('--num_epochs', type = int, default=5, help="train epoch")

BATCH_SIZE = 9
CROP_SIZE = 296
UPSCALE_FACTOR = 4
DIRPATH_TRAINDATA = "DIV_train_cropped"
DIRPATH_PRETRAIN = "Trained_model/Generator"
TOTAL_EPOCH = 5
grad_clip = None
lr = 1e-4

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu:0")

if __name__ == "__main__":

    opt = parser.parse_args()

    TOTAL_EPOCH = opt.num_epochs

    DIRPATH_TRAINDATA_HR = os.path.join(DIRPATH_TRAINDATA,"GT")
    DIRPATH_TRAINDATA_LR = os.path.join(DIRPATH_TRAINDATA,"LR_bicubic")

    train_dataset = Dataset_gen.Dataset_Train(hr_dirpath=DIRPATH_TRAINDATA_HR,lr_dirpath=DIRPATH_TRAINDATA_LR)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    generator = Generator()
    #generator = nn.DataParallel(generator)
    PRETRAINED_MODELPATH = os.path.join(DIRPATH_PRETRAIN, "generator_19th_model.pth")
    generator = utils.load_model(generator, filepath=PRETRAINED_MODELPATH, device=device)
    print("complete load model")

    image_discriminator = Discriminator(imagesize = (296,296))
    feat_discriminator = Discriminator(imagesize = (18,18))

    gen_optimizer = optim.Adam(generator.parameters(),lr= lr) #lr = 1e-4
    imgdis_optimizer = optim.Adam(image_discriminator.parameters(),lr = lr)
    featdis_optimizer = optim.Adam(feat_discriminator.parameters(),lr=lr)

    scheduler1 = optim.lr_scheduler.MultiStepLR(gen_optimizer,milestones=[3,5], gamma=0.1)
    scheduler2 = optim.lr_scheduler.MultiStepLR(imgdis_optimizer,milestones=[3,5], gamma=0.1)
    scheduler3 = optim.lr_scheduler.MultiStepLR(featdis_optimizer,milestones=[3,5], gamma=0.1)
    truncat_vgg = truncated_vgg()  # vgg(5,4) loss

    mseloss = nn.MSELoss()
    adversal_criterion = nn.BCEWithLogitsLoss()

    PSNR_train = np.zeros(TOTAL_EPOCH)
    Train_Gen_loss = np.zeros(TOTAL_EPOCH)
    Train_Dis_loss = np.zeros(TOTAL_EPOCH)
    train_len = len(train_dataloader)

    start_epoch = 0

    generator = nn.DataParallel(generator)
    image_discriminator = nn.DataParallel(image_discriminator)
    truncat_vgg = nn.DataParallel(truncat_vgg)

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
        for lr_image, hr_image in tqdm.tqdm(train_dataloader, bar_format="{l_bar}{bar:40}{r_bar}"):
        #for i, (input, hr_image) in enumerate(train_dataloader):
           # print("---batch {}---".format(i))
            target_list = np.array(hr_image)
            input_list = np.array(lr_image)
            """
            for i, target_image in enumerate(target_list):
                print("target {} : {}".format(i,np.array(target_image).shape))
            for i,input_image in enumerate(input):
                print("input {}: {}".format(i,np.array(input_image).shape))
            """
            lr_image, hr_image = lr_image.to(device), hr_image.to(device)

            imgdis_optimizer.zero_grad()
            gen_optimizer.zero_grad()
            featdis_optimizer.zero_grad()

            #generate fake hr images
            fake_hr = generator(lr_image)
           # generate vgg pathces
            fake_vgg_patch = truncat_vgg(fake_hr)/12.75
            real_vgg_patch = truncat_vgg(hr_image)/12.75

           # train image Discriminator
            img_fake_crimed = image_discriminator(fake_hr)
            img_real_crimed = image_discriminator(hr_image)

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
            #common_loss = mseloss(fake_hr,hr_image)

            gen_total_loss = perceptual_loss + 1e-3*(gen_img_score+gen_feat_score)
            Train_Gen_loss[epoch] += gen_total_loss

            gen_total_loss.backward()
            gen_optimizer.step()

            total_PSNR_train += mseloss(fake_hr,hr_image)
            #print("epoch {} training step : {}/{}".format(epoch+1, i + 1, train_len))

            imgdis_optimizer.requires_grad_(True)
            featdis_optimizer.requires_grad_(True)

        scheduler1.step()
        scheduler2.step()
        scheduler3.step()

        Train_Gen_loss[epoch] /= train_len
        PSNR_train[epoch] = total_PSNR_train/train_len
        print("train PSNR in epoch {} : {}".format(epoch+1,PSNR_train[epoch]))
     #   Train_Gen_loss[epoch] = Gen_loss_total / len(train_dataloader)
     #   Train_Dis_loss[epoch] = Dis_loss_total / len(train_dataloader)
     #   PSNR_train[epoch] = total_PSNR_train / len(train_dataloader)
     #   print("train PSNR : {}".format(total_PSNR_train / len(train_dataloader)))

        np.save("result_data/Generator/Train_Gen_loss.npy",Train_Gen_loss)
        #np.save("result_data/Discriminator/Train_Dis_loss.npy", Train_Dis_loss)
        np.save("result_data/PSNR_train.npy",PSNR_train)
        #np.save("result_data/PSNR_eval.npy",PSNR_eval)
    
        torch.save(generator.module.state_dict(), "Trained_model/post_Generator/generator_{}th_model.pth".format(epoch))
        torch.save(feat_discriminator.module.state_dict(), "Trained_model/Discriminator/discriminator_{}th_model.pth".format(epoch))
        torch.save(feat_discriminator.module.state_dict(), "Trained_model/Discriminator/discriminator_{}th_model.pth".format(epoch))

