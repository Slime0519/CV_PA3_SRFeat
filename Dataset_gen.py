import os, glob
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as torch_transform

from PIL import Image
from torch.utils.data import Dataset, DataLoader

class Dataset_Pretrain(Dataset):
    def __init__(self, dirpath, crop_size = 296, upscale_factor = 4, extension = '.jpg'):
        super(Dataset_Pretrain, self).__init__()
        self.imagelist = glob.glob(os.path.join(dirpath,"*"+extension))
      # self.cropsize = crop_size - (crop_size%upscale_factor)
        self.cropsize = crop_size
        self.hr_transform = hr_transform(self.cropsize)
        self.lr_transform = lr_transform(self.cropsize, upscale_factor=upscale_factor)

    def __getitem__(self, index):
        image = Image.open(self.imagelist[index]).convert("RGB")
      #  image = np.array(image)

        hr_image = self.hr_transform(image)
        lr_image = self.lr_transform(hr_image)
       # print("imagesize : {}".format(np.shape(image)))
        return lr_image, hr_image

    def __len__(self):
        return len(self.imagelist)

class Dataset_Train(Dataset):
    def __init__(self, hr_dirpath, lr_dirpath, extension = '.png'):
        super(Dataset_Train, self).__init__()
        self.hr_imagelist = glob.glob(os.path.join(hr_dirpath,"*"+extension))
        self.lr_imagelist = glob.glob(os.path.join(lr_dirpath,"*"+extension))
        self.totensor = torch_transform.Compose([
            torch_transform.ToTensor()
        ])
    def __getitem__(self, index):
        hr_image = Image.open(self.hr_imagelist[index]).convert("RGB")
        lr_image = Image.open(self.lr_imagelist[index]).convert("RGB")

        hr_image = self.totensor(hr_image)
        lr_image = self.totensor(lr_image)

        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_imagelist)

class Dataset_Validation(Dataset):
    def __init__(self, dirpath, crop_size = 296, upscale_factor = 4):
        super(Dataset_Validation, self).__init__()
        self.upscale_factor = upscale_factor
        self.crop_size = crop_size
        self.imagelist = glob.glob(os.path.join(dirpath,"*"))

    def __getitem__(self, index):
        image = Image.open(self.imagelist[index]).convert("RGB")
        #image = np.array(image)
        self.hr_transform = hr_transform(self.crop_size)
        self.lr_transform = lr_transform(self.crop_size, self.upscale_factor)

        hr_image = self.hr_transform(image)
        lr_image = self.lr_transform(hr_image)
      #  print("size of hr_image : {}".format(hr_image.shape))
      #  print("size of hr_image : {}".format(lr_image.shape))
        return lr_image, hr_image

    def __len__(self):
        return len(self.imagelist)

class Dataset_Test(Dataset):
    def __init__(self, dirpath):
        super(Dataset_Test, self).__init__()
       # self.upscale_factor = upscale_factor
        self.imagelist = glob.glob(os.path.join(dirpath, "*"))
        self.transform = torch_transform.ToTensor()
    def __getitem__(self, index):
        image = Image.open(self.imagelist[index]).convert("RGB")
        image = self.transform(image)

        return image

    def __len__(self):
        return len(self.imagelist)

def hr_transform(crop_size = 296):
    transform = torch_transform.Compose([
        torch_transform.ToTensor(),
        torch_transform.ToPILImage(),
        torch_transform.CenterCrop(crop_size),
        torch_transform.ToTensor()
    ])
    return transform

def lr_transform(crop_size = 296, upscale_factor = 4):
    transform = torch_transform.Compose([
        torch_transform.ToPILImage(),
        torch_transform.Resize(int(crop_size//upscale_factor),interpolation=Image.BICUBIC),
        torch_transform.ToTensor()
    ])
    return transform

if __name__ == "__main__":
    dirpath_train = "Dataset/Train"
    dirpath_vaild = "Dataset/Vaild"
    Test_Traindataset = Dataset_Train(dirpath= dirpath_train, crop_size= 96, upscale_factor= 4)
    Test_Vailddataset = Dataset_Validation(dirpath= dirpath_vaild, upscale_factor=4)

    Test_TraindataLoader = DataLoader(dataset=Test_Traindataset, batch_size=16, shuffle=True)
    Test_VailddataLoader = DataLoader(dataset=Test_Vailddataset, batch_size=1)

    for input, target in Test_TraindataLoader:
        print(input[0].shape)
        inputimage = np.array(input[0])
        print(inputimage)
        inputimage = np.transpose(inputimage,(1,2,0))
        plt.imshow(inputimage)
        plt.show()
        outputimage = np.array(target[0])
        outputimage = np.transpose(outputimage,(1,2,0))
        plt.imshow(outputimage)
        plt.show()
        break

    for input, target in Test_VailddataLoader:
        print(input[0].shape)
        inputimage = np.array(input[0])
        print(inputimage)
        inputimage = np.transpose(inputimage, (1, 2, 0))
        plt.imshow(inputimage)
        plt.show()
        outputimage = np.array(target[0])
        outputimage = np.transpose(outputimage, (1, 2, 0))
        plt.imshow(outputimage)
        plt.show()
        break
