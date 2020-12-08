import os
import glob
import re
import torch
import numpy as np

from PIL import Image

def load_model(model,filepath,device):
    model.load_state_dict(torch.load(filepath))
    model.to(device)
    return model

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





