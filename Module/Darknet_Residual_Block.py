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
from Module.Darknet_Convolutional import *

class Darknet_Residual_Block(nn.Module):
    def __init__(self,input_channel,size,amount):
        self.layers=[]
        for i in range(amount):
            self.layers.append(Darknet_Convolutional(input_channel,input_channel/2,1))
            self.layers.append(Darknet_Convolutional(input_channel/2,input_channel,3,padding=1))
        return

    def forward(self, input):
        res=input
        for i in range(len(self.layers)/2):
            ori=res
            res=self.layers[2*i](res)
            res=self.layers[2*i+1](res)
            res=ori+res
        return res