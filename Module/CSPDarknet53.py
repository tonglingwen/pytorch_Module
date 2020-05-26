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
from Module.Darknet53 import *

class CSP_Block(nn.Module):
    def __init__(self,layer,in_channels):
        super(CSP_Block, self).__init__()
        self.layer=layer
        self.transition0=Darknet_Convolutional(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
        #self.transition1=Darknet_Convolutional(in_channels=in_channels,out_channels=2*in_channels,kernel_size=3,stride=1,padding=1)

    def forward(self, input):
        size=int(input.size()[1]/2)
        part1=input[:,:size,:,:]
        part2=input[:,size:,:,:]
        part2 = self.layer(part2)
        part2 = self.transition0(part2)
        res = torch.cat((part1, part2), dim=1)
        #res=self.transition1(torch.cat((part1,part2),dim=-1))
        return res

class CSPDarknet53(nn.Module):
    def __init__(self):
        super(CSPDarknet53,self).__init__()
        self.conv0=Darknet_Convolutional(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.conv1=Darknet_Convolutional(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1)

        self.residual0=Darknet_Residual_Block(32)
        self.csp_block0=CSP_Block(self.residual0,32)

        self.conv2=Darknet_Convolutional(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)

        self.residual1 =nn.Sequential(
            Darknet_Residual_Block(64),
            Darknet_Residual_Block(64)
        )
        self.csp_block1 = CSP_Block(self.residual1, 64)

        self.conv3 = Darknet_Convolutional(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

        self.residual2 =nn.Sequential(
            Darknet_Residual_Block(128),
            Darknet_Residual_Block(128),
            Darknet_Residual_Block(128),
            Darknet_Residual_Block(128),
            Darknet_Residual_Block(128),
            Darknet_Residual_Block(128),
            Darknet_Residual_Block(128),
            Darknet_Residual_Block(128)
        )
        self.csp_block2 = CSP_Block(self.residual2, 128)

        self.conv4 = Darknet_Convolutional(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)

        self.residual3 =nn.Sequential(
            Darknet_Residual_Block(256),
            Darknet_Residual_Block(256),
            Darknet_Residual_Block(256),
            Darknet_Residual_Block(256),
            Darknet_Residual_Block(256),
            Darknet_Residual_Block(256),
            Darknet_Residual_Block(256),
            Darknet_Residual_Block(256),
        )
        self.csp_block3 = CSP_Block(self.residual3, 256)

        self.conv5 = Darknet_Convolutional(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)

        self.residual4 =nn.Sequential(
            Darknet_Residual_Block(512),
            Darknet_Residual_Block(512),
            Darknet_Residual_Block(512),
            Darknet_Residual_Block(512)
        )
        self.csp_block4 = CSP_Block(self.residual4, 512)

        self.fc=nn.Linear(1024,1000)
        return

    def forward(self, input):
        res=self.conv0(input)
        res = self.conv1(res)
        res=self.csp_block0(res)
        res = self.conv2(res)
        res = self.csp_block1(res)
        res = self.conv3(res)
        res = self.csp_block2(res)
        res = self.conv4(res)
        res =self.csp_block3(res)
        res = self.conv5(res)
        res =self.csp_block4(res)
        size=res.size()
        res=F.avg_pool2d(res,size[2],1)
        res=res.view(-1,size[1])
        res=self.fc(res)
        res=F.softmax(res,dim=-1)
        return res

net=CSPDarknet53()



summary(net, input_size=(3, 256, 256),device='cpu')
#input=torch.rand([1,3,256,256])
#net(input)