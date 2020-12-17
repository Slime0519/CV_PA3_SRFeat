import torch, tqdm, argparse, os
import Dataset_gen, utils

from torch.utils.data import DataLoader
from Models.Generator_128 import Generator
from Models.NotBN_Generator_128 import NotBN_Generator
from torchvision.utils import save_image
from piq import psnr

parser = argparse.ArgumentParser(description="SRFeat Validation Module")
parser.add_argument('--generator_version', type = str, default='post', help="specify version of generator wherther 'pretrain' or 'final'")
parser.add_argument('--pre_trained_epoch', type=int, default=0, help="epoch of trained model")
parser.add_argument('--dataset_name', type = str, default='BSD100', help = "name of dataset for test")

CROP_SIZE = 296
UPSCALE_FACTOR = 4
DIRPATH_Validation = "Dataset/valid"
DIRPATH_TRAINED = "Trained_model"
DIRPATH_SAVEIMAGE= "Valid_result/"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu:0")

if __name__ == "__main__":

    opt = parser.parse_args()
    generator_ver = opt.generator_version
    pretrained_epoch = opt.pre_trained_epoch
    dataset_name = opt.dataset_name

    pretrained_modelpath = utils.specify_generator_path(DIRPATH_TRAINED, generator_ver, pretrained_epoch)
    datasetpath = utils.specify_dataset_path(DIRPATH_Validation, dataset_name)
    savefolderpath = os.path.join(DIRPATH_SAVEIMAGE, generator_ver + "_" + dataset_name)

    if not os.path.exists(savefolderpath):
        os.mkdir(savefolderpath)

    valid_dataset = Dataset_gen.Dataset_Validation(dirpath=datasetpath, crop_size=CROP_SIZE,
                                                        upscale_factor=UPSCALE_FACTOR)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=2,
                                       pin_memory=True)


#    print(pretrained_modelpath)
    generator = Generator()
    if generator_ver == "NotBN_pretrain" or generator_ver == "NotBN":
        generator = NotBN_Generator()

    validation_PSNR =0
   
    if generator_ver == "pretrain":
        generator = utils.load_model(generator, filepath = pretrained_modelpath)
    else:
        generator.load_state_dict(torch.load(pretrained_modelpath),strict=True)
    generator = generator.to(device)

    accum_psnr = 0
    generator.eval()
    print("validate about dataset {}".format(dataset_name))

    num_image = 0
    for lr_image, hr_image in tqdm.tqdm(valid_dataloader, bar_format="{l_bar}{bar:40}{r_bar}"):

        lr_image, hr_image = lr_image.to(device), hr_image.to(device)

        # generate fake hr images
        fake_hr = generator(lr_image)

        temp_fake = torch.clamp(fake_hr, min=0, max=1)
        accum_psnr += psnr(temp_fake, hr_image)
        #temp_mse = None
        #temp_fake = None
        #fake_hr = None

        save_image(temp_fake[0],  os.path.join(savefolderpath, "image{}.png".format(num_image)))
        num_image +=1

    validation_PSNR = accum_psnr / len(valid_dataloader)
    print("average PSNR about dataset {}: {}".format(dataset_name, validation_PSNR))
