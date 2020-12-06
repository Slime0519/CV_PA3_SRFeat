import torch

import Model
import Dataset_gen
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as pltimage
import numpy as np
import os
import cv2
from Test import regularization_image
from PIL import Image

savedir = "Result_image"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def compare_image(image1, image2, image1_epoch, image2_epoch, save = False, num =0):
    fig = plt.figure()
    rows = 1
    columns = 2
    fig.suptitle("compare two model epoch {} and {}".format(image1_epoch, image2_epoch))
    bicubic_grid = fig.add_subplot(rows, columns, 1)
    bicubic_grid.imshow(image1)
    bicubic_grid.set_title("epoch {} model".format(image1_epoch))
    bicubic_grid.axis("off")

    srgan_grid= fig.add_subplot(rows, columns, 2)
    srgan_grid.imshow(image2)
    srgan_grid.set_title("epoch {} model".format(image2_epoch))
    srgan_grid.axis("off")
    if save:
        plt.savefig(os.path.join(savedir,"compare","compare_{}_{}".format(image1_epoch,image2_epoch),"image_{}".format(num)),dpi=500)
    else:
        plt.show()

if __name__ == "__main__":
    testset_dirpath = "Dataset/Test"
    testset_name = "BSDS300"

    model_dirpath = "Trained_model"
    model_epoch1 = 50
    model_epoch2 = 100

    gen_model_100 = Model.Generator()
    gen_model_300 = Model.Generator()
    Test_Dataset = Dataset_gen.Dataset_Test(dirpath=os.path.join(testset_dirpath, testset_name))
    Test_Dataloader = DataLoader(dataset=Test_Dataset, shuffle=False, batch_size=1, num_workers=0)

    gen_model_100.load_state_dict(
        torch.load(os.path.join(model_dirpath, "Generator", "generator_{}th_model.pth".format(model_epoch1 - 1))))
    gen_model = gen_model_100.to(device)
    gen_model.eval()

    gen_model_300.load_state_dict(
        torch.load(os.path.join(model_dirpath, "Generator", "generator_{}th_model.pth".format(model_epoch2 - 1))))
    gen_model = gen_model_300.to(device)
    gen_model.eval()

    for i, input in enumerate(Test_Dataloader):
        input = input.to(device)
        output1 = gen_model_100(input)
        output_image1 = np.array(output1.cpu().detach())
        output_image1 = output_image1.squeeze()
        output_image1 = np.transpose(output_image1, (1, 2, 0))
        regularized_output_image1 = regularization_image(output_image1)
        regularized_output_image1 = (regularized_output_image1 * 255).astype(np.uint8)

        output2 = gen_model_300(input)
        output_image2 = np.array(output2.cpu().detach())
        output_image2 = output_image2.squeeze()
        output_image2 = np.transpose(output_image2, (1, 2, 0))
        regularized_output_image2 = regularization_image(output_image2)
        regularized_output_image2 = (regularized_output_image2 * 255).astype(np.uint8)

        compare_image(image1=regularized_output_image1, image2=regularized_output_image2, image1_epoch=model_epoch1, image2_epoch=model_epoch2, save=True, num=i + 1)

