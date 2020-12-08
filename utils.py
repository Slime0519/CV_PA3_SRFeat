import os
import glob
import re
import torch
import numpy as np

from PIL import Image

def load_module(dirpath):
    filelist = glob.glob(os.path.join(dirpath,"*.pth"))

    listlen = len(filelist)
    sorted_filelist = [0 for _ in range(listlen)]
    for filename in filelist:
        epoch = int(*re.findall("\d+",filename))
      #  print(epoch)
        sorted_filelist[epoch] = filename

    return listlen, sorted_filelist[-1]

def randomcrop(image, crop_size):
    size = image.shape
    random_x, random_y = np.random.randint(0,size[0]-crop_size), np.random.randint(0,size[1]-crop_size)
    cropped_image = image[random_x:random_x+crop_size,random_y:random_y+crop_size,:]

    return cropped_image

def remove_small_images(datasetpath, minimum=296):
    Dataset_PATH = datasetpath
    imagelist = os.listdir(Dataset_PATH)
    
    length = len(imagelist)
    removelist = []
    print("investigate {} images".format(length))

    for i,imagename in enumerate(imagelist):
        imagepath = os.path.join(Dataset_PATH,imagename)
        image = Image.open(imagepath)
        imagesize = np.array(image).shape
        #print("investigate {}th images".format(i+1))
        #if i+1 %(length//10) ==0:
        #    print("investigate {}th images : {}%, number of small images : {}".format(i+1,float(i)/length,))
        #if (imagesize[2]==1):
        #    removelist.append(imagepath)
        
        if (imagesize[0] <minimum) or (imagesize[1]<minimum):
            removelist.append(imagepath)
        if i+1 %(length//10) ==0:
            print("investigate {}th images : {}%, number of small images : {}".format(i+1,float(i)/length,len(removelist)))
    
    for imagepath in removelist:
        os.remove(imagepath)

    print("number of small images : {}".format(len(removelist)))
    print("ratio of removed images : {}/{} = {}".format(len(removelist),len(imagelist),len(removelist)/float(len(imagelist))))





