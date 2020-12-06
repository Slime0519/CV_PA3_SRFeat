import glob
from PIL import Image
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="SRGAN Training Module")
parser.add_argument('--dirpath', type = str, default="Dataset/Train", help = "path of pretrained models")
parser.add_argument('--crop_size', type = int, default=96, help = "threshold of crop size")

if __name__ == "__main__":
    opt = parser.parse_args()
    removelist = []
    imagelist = glob.glob(os.path.join(opt.dirpath,"*.jpg"))

    for i,imagename in enumerate(imagelist):
        image = Image.open(imagename)
        np_image = np.array(image)
        print("check {}th image of total {} image".format(i,len(imagelist)))
        if (np_image.shape[0] < opt.crop_size) or (np_image.shape[1] < opt.crop_size):
            os.remove(imagename)
            removelist.append(imagename)
            print("file remove : {}".format(imagename))

    if removelist == []:
        print("all file size over {}x{}".format(opt.crop_size,opt.crop_size))
    else :
        print("----remove file list----")
        print(removelist)
