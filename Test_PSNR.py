import torch
import torch.nn as nn
import torch.utils as utils
import Model
import Dataset_gen
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as pltimage
import numpy as np
import os
import cv2
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim
from Test import regularization_image

savedir = "Result_image"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def regularization_image(image):
    min = np.min(image)
    temp_image = image - min

    max = np.max(temp_image)
    temp_image = temp_image / max

    return temp_image


def compare_image(bicubic_image, srgan_image, epoch, save=False, num=0):
    fig = plt.figure()
    rows = 1
    columns = 2
    fig.suptitle("epoch {}".format(epoch))
    bicubic_grid = fig.add_subplot(rows, columns, 1)
    bicubic_grid.imshow(bicubic_image)
    bicubic_grid.set_title("bicubic interpolation")
    bicubic_grid.axis("off")

    srgan_grid = fig.add_subplot(rows, columns, 2)
    srgan_grid.imshow(srgan_image)
    srgan_grid.set_title("SRGAN")
    srgan_grid.axis("off")

    if save:
        plt.savefig(os.path.join(savedir, "compare", "epoch_{}".format(epoch), "image_{}".format(num)), dpi=500)
    else:
        plt.show()


if __name__ == "__main__":
    testset_dirpath = "Dataset/Test"
    testset_name = "BSDS300"

    model_dirpath = "Trained_model"
    model_epoch = 300

    gen_model = Model.Generator()
    Test_Dataset = Dataset_gen.Dataset_Vaild(dirpath=os.path.join(testset_dirpath, testset_name))
    Test_Dataloader = DataLoader(dataset=Test_Dataset, shuffle=True, batch_size=1, num_workers=0)

    gen_model.load_state_dict(
        torch.load(os.path.join(model_dirpath, "Generator", "generator_{}th_model.pth".format(model_epoch - 1))))
    gen_model = gen_model.to(device)
    gen_model.eval()

    for i, (input,original_input_image) in enumerate(Test_Dataloader):
        input = input.to(device)

        output = gen_model(input)
        output_image = np.array(output.cpu().detach())
        output_image = output_image.squeeze()
        output_image = np.transpose(output_image, (1, 2, 0))
        regularized_output_image = regularization_image(output_image)
        regularized_output_image = (regularized_output_image * 255).astype(np.uint8)

        input_temp = np.array(input.cpu().detach())
        input_bicubic = cv2.resize(np.transpose(np.squeeze(input_temp), (1, 2, 0)), dsize=(0, 0), fx=4, fy=4,
                                   interpolation=cv2.INTER_CUBIC)
        regularized_input_image = regularization_image(input_bicubic)
        regularized_input_image = (regularized_input_image * 255).astype(np.uint8)

        original_input_image =np.array(original_input_image).squeeze()
        original_input_image = np.transpose(original_input_image,(1,2,0))
        regularized_original_image = regularization_image(original_input_image)

        plt.imshow(regularized_original_image)
        plt.show()
        # PNG Image 저장
        PIL_Input_Image = Image.fromarray(regularized_input_image).convert('RGB')
        # PIL_Input_Image.save("Result_image/bicubic/epoch{}_image{}.png".format(model_epoch,i))
        PIL_Input_Image.save("Result_image/bicubic/epoch{}_image{}.png".format(model_epoch,i))  # save large size image

        PIL_output_Image = Image.fromarray(regularized_output_image).convert('RGB')
        # PIL_output_Image.save("Result_image/srgan/epoch{}_image{}.png".format(model_epoch, i))
        PIL_output_Image.save("Result_image/srgan/epoch{}_image{}.png".format(model_epoch,i))

        print("PSNR : {}".format(compare_psnr(regularized_input_image, regularized_original_image)))
        print("ssim : {}".format(compare_ssim(regularized_input_image, regularized_original_image,multichannel=True)))

    #  compare_image(bicubic_image=regularized_input_image, srgan_image=regularized_output_image, epoch=model_epoch,
    #               save=True, num=i+1)
    #  if i % 10 == 0:
    #    compare_image(bicubic_image=regularized_input_image,srgan_image=regularized_output_image,epoch=model_epoch,save = False)
    # pltimage.imsave(os.path.join(savedir,"my_{}th_image.jpg".format(i)),regularized_output_image)

    PSNR_eval = np.load("result_data/PSNR_eval.npy")
    PSNR_Train = np.load("result_data/PSNR_train.npy")
    Train_Dis_loss = np.load("result_data/Train_Dis_loss.npy")
    Train_Gen_loss = np.load("result_data/Train_Gen_loss.npy")

    Num_Epoch = len(PSNR_eval)
    x = range(Num_Epoch)
    y_eval = PSNR_eval
    y_train = PSNR_Train
    plt.plot(x, y_train)
    plt.plot(x, y_eval)
    plt.legend(['train PSNR', 'evaluation PSNR'])
    plt.title("average PSNR at train and evaluation step")
    #  plt.savefig("result_data/average_PSNR.png")
    plt.show()

    dis_loss = Train_Dis_loss
    gen_loss = Train_Gen_loss

    plt.plot(x, dis_loss)
    plt.plot(x, gen_loss)
    plt.legend(['discriminator loss', 'generator loss'])
    plt.title("average loss of generator and discriminator loss at training step")
    # plt.savefig("result_data/average_loss.png")
    plt.show()



