import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
from graphviz import Digraph
from Module.visualize import make_dot
import math
import sys
import numpy as np
import time
import data.voc0712 as voc
import torch.utils.data as data
from Module.Darknet_Residual_Block import *
from Module.Darknet_Convolutional import *

class Darknet_53(nn.Module):
    def __init__(self):

        self.conv0=Darknet_Convolutional(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.conv1=Darknet_Convolutional(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.residual0=Darknet_Residual_Block(64,1)
        self.conv2=Darknet_Convolutional(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
        self.residual1 = Darknet_Residual_Block(128, 2)
        self.conv3 = Darknet_Convolutional(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.residual2 = Darknet_Residual_Block(256, 8)
        self.conv4 = Darknet_Convolutional(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.residual3 = Darknet_Residual_Block(512, 8)
        self.conv5 = Darknet_Convolutional(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.residual4 = Darknet_Residual_Block(1024, 4)
        return

    def forward(self, input):
        res=self.conv0(input)
        res = self.conv1(res)
        res = self.residual0(res)
        res = self.conv2(res)
        res = self.residual1(res)
        res = self.conv3(res)
        res = self.residual2(res)
        res = self.conv4(res)
        res = self.residual3(res)
        res = self.conv5(res)
        res = self.residual4(res)
        return res