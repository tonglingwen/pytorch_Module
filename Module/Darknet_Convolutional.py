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

class Darknet_Convolutional(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.batchnormal=nn.BatchNorm2d(out_channels)

        return

    def forward(self, input):
        res=F.leaky_relu(self.batchnormal(self.conv(input)),negative_slope=0.1)
        return res