import os, glob, torch
import numpy as np

from PIL import Image

def load_model(model,filepath):
    state_dict = torch.load(filepath)
    new_state_dict= {}
    oldkeys = state_dict.copy().keys()

    #eliminate prefix module. concated by using nn.Dataparallel function
    for key in oldkeys:
        prefix_loc = key.find('module.')
        if prefix_loc == 0:
            newkey = key.replace("module.","",1)
            new_state_dict[newkey] = state_dict.pop(key)
    model.load_state_dict(new_state_dict)

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
        
        if (imagesize[0] <minimum) or (imagesize[1]<minimum):
            removelist.append(imagepath)
        if i+1 %(length//10) ==0:
            print("investigate {}th images : {}%, number of small images : {}".format(i+1,float(i)/length,len(removelist)))
    
    for imagepath in removelist:
        os.remove(imagepath)

    print("number of small images : {}".format(len(removelist)))
    print("ratio of removed images : {}/{} = {}".format(len(removelist),len(imagelist),len(removelist)/float(len(imagelist))))

def specify_generator_path(Basedir, version,epoch):
    if version == 'pretrain':
        gen_path = os.path.join(Basedir,"Generator","generator_{}th_model.pth".format(epoch-1))
    elif version== 'NotBN_pretrain':
        gen_path = os.path.join(Basedir,"NotBN_Generator","generator_{}th_model.pth".format(epoch-1))
    else:
        gen_path = os.path.join(Basedir,"post_Generator","generator_{}th_model.pth".format(epoch-1))
    return gen_path

def specify_dataset_path(Basedir,setname):
    datasetpath_list = glob.glob(os.path.join(Basedir, "*"))
    #print("setname : {}".format(setname))
    index = None
    for i, path in enumerate(datasetpath_list): 
        #print(path)
        if path.split('/')[-1] == setname:
            index = i
    
    return datasetpath_list[index]







