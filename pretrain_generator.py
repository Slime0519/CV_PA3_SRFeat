import torch, argparse, os, tqdm

import numpy as np
import torch.optim as optim
import torch.nn as nn
import Dataset_gen

from torch.utils.data import DataLoader
from Models.Generator_128 import Generator

parser = argparse.ArgumentParser(description="SRFeat Training Module")
parser.add_argument('--pre_trained', type=int, default=0, help="path of pretrained models")
parser.add_argument('--num_epochs', type=int, default=20, help="train epoch")

BATCH_SIZE = 9
CROP_SIZE = 296
UPSCALE_FACTOR = 4
DIRPATH_TRAIN = "Dataset/train/COCO/train2017"
DIRPATH_PRETRAIN = "Trained_model/Generator"

TOTAL_EPOCHS = 20
lr = 1e-4

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu:0")

if __name__ == "__main__":

    opt = parser.parse_args()

    PRETRAINED_EPOCH = opt.pre_trained
    TOTAL_EPOCHS = opt.num_epochs

    loss_array_Train = np.zeros(TOTAL_EPOCHS)
    PSNR_array_Train = np.zeros(TOTAL_EPOCHS)
    PSNR_array_Vaild = np.zeros(TOTAL_EPOCHS)

    train_dataset = Dataset_gen.Dataset_Pretrain(dirpath=DIRPATH_TRAIN, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory= True)

    generator = Generator()
    gen_optimizer = optim.Adam(generator.parameters(), lr=lr)  # lr = 1e-4
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, T_max = TOTAL_EPOCHS//2, eta_min=1e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(gen_optimizer, milestones=[10, 15], gamma=0.1)

    mseloss = nn.MSELoss()

    PSNR_train = np.zeros(TOTAL_EPOCHS)
    Train_Gen_loss = np.zeros(TOTAL_EPOCHS)
    datasize = len(train_dataloader)

    PRETRAINED_MODELPATH = os.path.join(DIRPATH_PRETRAIN,"generator_{}th_model.pth".format(PRETRAINED_EPOCH-1))
    generator = nn.DataParallel(generator)
    generator = generator.to(device)

    start_epoch = 0

    if PRETRAINED_EPOCH>0:
        start_epoch = PRETRAINED_EPOCH
        for i in range(min(PRETRAINED_EPOCH,TOTAL_EPOCHS//2)):
            scheduler.step()

    state_dict= {}
    np.save("result_data/pretrain/test.npy",PSNR_train)

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        # prepare training
        generator.train()

        total_MSE_train = 0
        accum_psnr = 0

        print("----epoch {}/{}----".format(epoch + 1, TOTAL_EPOCHS))
        for lr_image,hr_image in tqdm.tqdm(train_dataloader, bar_format="{l_bar}{bar:40}{r_bar}"):
            target_list = np.array(hr_image)
            
            input_list = np.array(lr_image)

            lr_image, hr_image = lr_image.to(device), hr_image.to(device)
            gen_optimizer.zero_grad()

            # generate fake hr images
            fake_hr = generator(lr_image)

            pretrain_loss = mseloss(fake_hr,hr_image)

            total_MSE_train += pretrain_loss
            accum_psnr += 10 * torch.log10(1/pretrain_loss)
            #accum_psnr += temp_psnr    #demand too much gpu memory

            pretrain_loss.backward()
            gen_optimizer.step()


        scheduler.step()

        PSNR_train[epoch] = accum_psnr/datasize
        Train_Gen_loss[epoch] = total_MSE_train/datasize

        print("average PSNR : {} | MSE : {}".format(PSNR_train[epoch],Train_Gen_loss[epoch]))

        torch.save(generator.module.state_dict(), "Trained_model/Generator/generator_{}th_model.pth".format(epoch))
        np.save("result_data/pretrain/PSNR_{}_to_{}.npy".format(start_epoch,epoch),PSNR_train)
        np.save("result_data/pretrain/Train_Gen_loss_{}_to_{}.npy".format(start_epoch,epoch),Train_Gen_loss)
