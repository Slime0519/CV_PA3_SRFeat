import torch, tqdm, argparse, os
import Dataset_gen, utils

from Models.Generator_128 import Generator
from torch.utils.data import DataLoader
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description="SRFeat Test Module")
parser.add_argument('--generator_version', type = str, default='post', help="specify version of generator wherther 'pretrain' or 'post'")
parser.add_argument('--pre_trained_epoch', type=int, default=0, help="epoch of trained model")
parser.add_argument('--dataset_name', type = str, default='BSD100', help = "name of dataset for test")

UPSCALE_FACTOR = 4
DIRPATH_Test = "Dataset/valid"
DIRPATH_TRAINED = "Trained_model"
DIRPATH_TESTIMAGE = "Test_result"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu:0")

if __name__ == "__main__":

    opt = parser.parse_args()

    testver = opt.generator_version
    pretrained_epoch = opt.pre_trained_epoch
    dataset_name = opt.dataset_name

    pretrained_modelpath = utils.specify_generator_path(DIRPATH_TRAINED,testver,pretrained_epoch)
    datasetpath = utils.specify_dataset_path(DIRPATH_TESTIMAGE, dataset_name)

    test_dataset = Dataset_gen.Dataset_Test(dirpath=datasetpath)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)


    generator = Generator()
    generator = utils.load_model(generator, filepath = pretrained_modelpath,device =device)
    generator = generator.to('cpu')

    generator.eval()

    specified_index = 1
    imagenum = 0
    print("upscale image about dataset {}".format(dataset_name))
    for lr_image in tqdm.tqdm(test_dataloader, bar_format="{l_bar}{bar:40}{r_bar}"):
        # generate fake hr images
        fake_hr = generator(lr_image)
        fake_hr = torch.clamp(fake_hr, min=0, max=1)

        fake_hr = fake_hr[0]
        save_image(fake_hr,os.path.join(DIRPATH_TESTIMAGE,testver+dataset_name, "image{}.png".format(imagenum)))
        imagenum += 1