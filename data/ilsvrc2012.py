"""VOC Dataset Classes
Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
Updated by: Ellis Brown, Max deGroot
"""
#from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import types
from numpy import random
from PIL import *
import torchvision.transforms as transforms


class ILSVRC2012DatasetValidation(data.Dataset):
    """VOC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
#BaseTransform()
    def __init__(self, root):
        self.root = root
        # self.size=size
        # self.mean=mean
        self.ids = list()
        self.valpath=self.root+"/ILSVRC2012_img_val/"
        self.path=self.root+"/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"

        self.scale=transforms.Resize([256,256])
        self.center=transforms.CenterCrop(224)
        self.totensor=transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])



        for root, dirs, files in os.walk(self.valpath):
            for file in files:
               self.ids.append([self.valpath+file])
        f = open(self.path, 'r')
        i=0
        for line in f:
            self.ids[i].append(int(line))
            i=i+1
        print("load image mount is:",i)


    def __getitem__(self, index):
        im, gt= self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = img_id[1]
        img = Image.open(img_id[0])
        t=img.mode
        img=self.totensor(self.center(self.scale(img)))
        if t=='L':
            img=img.expand([3,img.shape[1],img.shape[1]])
        img=self.normalize(img)
        return img, target
        # return torch.from_numpy(img), target, height, width