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
                 bias=False, padding_mode='zeros'):
        super(Darknet_Convolutional,self).__init__()
        self.conv=nn.Sequential(
                                nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode),
                                nn.BatchNorm2d(out_channels)
                            )

    def forward(self, input):
        res=F.leaky_relu(self.conv(input),negative_slope=0.1)
        return res


class Darknet_Residual_Block(nn.Module):
    def __init__(self,input_channel):
        super(Darknet_Residual_Block,self).__init__()
        self.resblock=nn.Sequential(
                        Darknet_Convolutional(in_channels=input_channel,out_channels= int(input_channel/2),kernel_size= 1),
                        Darknet_Convolutional(in_channels=int(input_channel / 2), out_channels=input_channel, kernel_size=3,padding=1)
                    )


    def forward(self, input):
        res=input
        ori=res
        res=self.resblock(res)
        res=ori+res
        return res



class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53,self).__init__()
        self.conv0=Darknet_Convolutional(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.conv1=Darknet_Convolutional(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.residual0=Darknet_Residual_Block(64)
        self.conv2=Darknet_Convolutional(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
        self.residual1 =nn.Sequential(
            Darknet_Residual_Block(128),
            Darknet_Residual_Block(128)
        )
        self.conv3 = Darknet_Convolutional(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.residual2 =nn.Sequential(
            Darknet_Residual_Block(256),
            Darknet_Residual_Block(256),
            Darknet_Residual_Block(256),
            Darknet_Residual_Block(256),
            Darknet_Residual_Block(256),
            Darknet_Residual_Block(256),
            Darknet_Residual_Block(256),
            Darknet_Residual_Block(256)
        )
        self.conv4 = Darknet_Convolutional(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.residual3 =nn.Sequential(
            Darknet_Residual_Block(512),
            Darknet_Residual_Block(512),
            Darknet_Residual_Block(512),
            Darknet_Residual_Block(512),
            Darknet_Residual_Block(512),
            Darknet_Residual_Block(512),
            Darknet_Residual_Block(512),
            Darknet_Residual_Block(512),
        )
        self.conv5 = Darknet_Convolutional(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.residual4 =nn.Sequential(
            Darknet_Residual_Block(1024),
            Darknet_Residual_Block(1024),
            Darknet_Residual_Block(1024),
            Darknet_Residual_Block(1024)
        )

        self.fc=nn.Linear(1024,1000)
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
        size=res.size()
        res=F.avg_pool2d(res,size[2],1)
        res=res.view(-1,size[1])
        res=self.fc(res)
        res=F.softmax(res,dim=-1)
        return res

# net=Darknet53()
#
# torch.save(net.state_dict(), "_ssd_par.pth")
#
# summary(net, input_size=(3, 256, 256),device='cpu')
#input=torch.rand([1,3,256,256])
#net(input)