import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

'''
input size:224x224
'''

class ResNet34(nn.Module):

    def __init__(self,channel=3,classes=1000):
        super(ResNet34, self).__init__()
        self.channel=channel
        self.classes=classes
        self.conv1=nn.Conv2d(self.channel,64,7,2,3)
        self.batchnor1=nn.BatchNorm2d(64)

        self.conv2_1=nn.Conv2d(64,64,1)
        self.batchnor2_1=nn.BatchNorm2d(64)
        self.conv3_1 = nn.Conv2d(64, 64, 3,1,1)
        self.batchnor3_1 = nn.BatchNorm2d(64)
        self.conv4_1 = nn.Conv2d(64, 256, 1)
        self.batchnor4_1 = nn.BatchNorm2d(256)
        self.node1_conv_1 = nn.Conv2d(64, 256, 1)
        self.node1_batchnor_1 = nn.BatchNorm2d(256)

        self.conv1_2=nn.Conv2d(256,64,1)
        self.batchnor1_2=nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3,1,1)
        self.batchnor2_2 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 256, 1)
        self.batchnor3_2 = nn.BatchNorm2d(256)

        self.conv1_3=nn.Conv2d(256,64,1)
        self.batchnor1_3=nn.BatchNorm2d(64)
        self.conv2_3 = nn.Conv2d(64, 64, 3,1,1)
        self.batchnor2_3 = nn.BatchNorm2d(64)
        self.conv3_3 = nn.Conv2d(64, 256, 1)
        self.batchnor3_3 = nn.BatchNorm2d(256)

        self.conv1_4=nn.Conv2d(256,128,1,2)
        self.batchnor1_4=nn.BatchNorm2d(128)
        self.conv2_4 = nn.Conv2d(128, 128, 3,1,1)
        self.batchnor2_4 = nn.BatchNorm2d(128)
        self.conv3_4 = nn.Conv2d(128, 512, 1)
        self.batchnor3_4 = nn.BatchNorm2d(512)
        self.node1_conv_4 = nn.Conv2d(256, 512, 1,2)
        self.node1_batchnor_4 = nn.BatchNorm2d(512)

        self.conv1_5=nn.Conv2d(512,128,1)
        self.batchnor1_5=nn.BatchNorm2d(128)
        self.conv2_5 = nn.Conv2d(128, 128, 3,1,1)
        self.batchnor2_5 = nn.BatchNorm2d(128)
        self.conv3_5 = nn.Conv2d(128, 512, 1)
        self.batchnor3_5 = nn.BatchNorm2d(512)

        self.conv1_6=nn.Conv2d(512,128,1)
        self.batchnor1_6=nn.BatchNorm2d(128)
        self.conv2_6 = nn.Conv2d(128, 128, 3,1,1)
        self.batchnor2_6 = nn.BatchNorm2d(128)
        self.conv3_6 = nn.Conv2d(128, 512, 1)
        self.batchnor3_6 = nn.BatchNorm2d(512)

        self.conv1_7=nn.Conv2d(512,128,1)
        self.batchnor1_7=nn.BatchNorm2d(128)
        self.conv2_7 = nn.Conv2d(128, 128, 3,1,1)
        self.batchnor2_7 = nn.BatchNorm2d(128)
        self.conv3_7 = nn.Conv2d(128, 512, 1)
        self.batchnor3_7 = nn.BatchNorm2d(512)

        self.conv1_8=nn.Conv2d(512,256,1,2)
        self.batchnor1_8=nn.BatchNorm2d(256)
        self.conv2_8 = nn.Conv2d(256, 256, 3,1,1)
        self.batchnor2_8 = nn.BatchNorm2d(256)
        self.conv3_8 = nn.Conv2d(256, 1024, 1)
        self.batchnor3_8 = nn.BatchNorm2d(1024)
        self.node1_conv_8 = nn.Conv2d(512, 1024, 1,2)
        self.node1_batchnor_8 = nn.BatchNorm2d(1024)

        self.conv1_9=nn.Conv2d(1024,256,1)
        self.batchnor1_9=nn.BatchNorm2d(256)
        self.conv2_9 = nn.Conv2d(256, 256, 3,1,1)
        self.batchnor2_9 = nn.BatchNorm2d(256)
        self.conv3_9 = nn.Conv2d(256, 1024, 1)
        self.batchnor3_9 = nn.BatchNorm2d(1024)

        self.conv1_10=nn.Conv2d(1024,256,1)
        self.batchnor1_10=nn.BatchNorm2d(256)
        self.conv2_10 = nn.Conv2d(256, 256, 3,1,1)
        self.batchnor2_10 = nn.BatchNorm2d(256)
        self.conv3_10 = nn.Conv2d(256, 1024, 1)
        self.batchnor3_10 = nn.BatchNorm2d(1024)

        self.conv1_11=nn.Conv2d(1024,256,1)
        self.batchnor1_11=nn.BatchNorm2d(256)
        self.conv2_11 = nn.Conv2d(256, 256, 3,1,1)
        self.batchnor2_11 = nn.BatchNorm2d(256)
        self.conv3_11 = nn.Conv2d(256, 1024, 1)
        self.batchnor3_11 = nn.BatchNorm2d(1024)

        self.conv1_12=nn.Conv2d(1024,256,1)
        self.batchnor1_12=nn.BatchNorm2d(256)
        self.conv2_12 = nn.Conv2d(256, 256, 3,1,1)
        self.batchnor2_12 = nn.BatchNorm2d(256)
        self.conv3_12 = nn.Conv2d(256, 1024, 1)
        self.batchnor3_12 = nn.BatchNorm2d(1024)

        self.conv1_13=nn.Conv2d(1024,256,1)
        self.batchnor1_13=nn.BatchNorm2d(256)
        self.conv2_13 = nn.Conv2d(256, 256, 3,1,1)
        self.batchnor2_13 = nn.BatchNorm2d(256)
        self.conv3_13 = nn.Conv2d(256, 1024, 1)
        self.batchnor3_13 = nn.BatchNorm2d(1024)

        self.conv1_14=nn.Conv2d(1024,512,1,2)
        self.batchnor1_14=nn.BatchNorm2d(512)
        self.conv2_14 = nn.Conv2d(512, 512, 3,1,1)
        self.batchnor2_14 = nn.BatchNorm2d(512)
        self.conv3_14 = nn.Conv2d(512, 2048, 1)
        self.batchnor3_14 = nn.BatchNorm2d(2048)
        self.node1_conv_14 = nn.Conv2d(1024, 2048, 1,2)
        self.node1_batchnor_14 = nn.BatchNorm2d(2048)

        self.conv1_15=nn.Conv2d(2048,512,1)
        self.batchnor1_15=nn.BatchNorm2d(512)
        self.conv2_15 = nn.Conv2d(512, 512, 3,1,1)
        self.batchnor2_15 = nn.BatchNorm2d(512)
        self.conv3_15 = nn.Conv2d(512, 2048, 1)
        self.batchnor3_15 = nn.BatchNorm2d(2048)

        self.conv1_16=nn.Conv2d(2048,512,1)
        self.batchnor1_16=nn.BatchNorm2d(512)
        self.conv2_16 = nn.Conv2d(512, 512, 3,1,1)
        self.batchnor2_16 = nn.BatchNorm2d(512)
        self.conv3_16 = nn.Conv2d(512, 2048, 1)
        self.batchnor3_16 = nn.BatchNorm2d(2048)

        self.fc=nn.Linear(2048,self.classes);
        print("__init__")

    def forward(self, input):
        re=F.relu(self.batchnor1(self.conv1(input)))
        node1=F.max_pool2d(re,3,2,1)

        re=F.relu(self.batchnor2_1(self.conv2_1(node1)))
        re = F.relu(self.batchnor3_1(self.conv3_1(re)))
        re=self.batchnor4_1(self.conv4_1(re))
        node1=self.node1_batchnor_1(self.node1_conv_1(node1))
        node2=F.relu(node1+re)

        re=F.relu(self.batchnor1_2(self.conv1_2(node2)))
        re = F.relu(self.batchnor2_2(self.conv2_2(re)))
        re=self.batchnor3_2(self.conv3_2(re))
        node3=F.relu(node2+re)

        re=F.relu(self.batchnor1_3(self.conv1_3(node3)))
        re = F.relu(self.batchnor2_3(self.conv2_3(re)))
        re=self.batchnor3_3(self.conv3_3(re))
        node4=F.relu(node3+re)

        re=F.relu(self.batchnor1_4(self.conv1_4(node4)))
        re = F.relu(self.batchnor2_4(self.conv2_4(re)))
        re=self.batchnor3_4(self.conv3_4(re))
        node4=self.node1_batchnor_4(self.node1_conv_4(node4))
        node5=F.relu(node4+re)

        re=F.relu(self.batchnor1_5(self.conv1_5(node5)))
        re = F.relu(self.batchnor2_5(self.conv2_5(re)))
        re=self.batchnor3_5(self.conv3_5(re))
        node6=F.relu(node5+re)

        re=F.relu(self.batchnor1_6(self.conv1_6(node6)))
        re = F.relu(self.batchnor2_6(self.conv2_6(re)))
        re=self.batchnor3_6(self.conv3_6(re))
        node7=F.relu(node6+re)

        re=F.relu(self.batchnor1_7(self.conv1_7(node7)))
        re = F.relu(self.batchnor2_7(self.conv2_7(re)))
        re=self.batchnor3_7(self.conv3_7(re))
        node8=F.relu(node7+re)

        re=F.relu(self.batchnor1_8(self.conv1_8(node8)))
        re = F.relu(self.batchnor2_8(self.conv2_8(re)))
        re=self.batchnor3_8(self.conv3_8(re))
        node8=self.node1_batchnor_8(self.node1_conv_8(node8))
        node9=F.relu(node8+re)

        re=F.relu(self.batchnor1_9(self.conv1_9(node9)))
        re = F.relu(self.batchnor2_9(self.conv2_9(re)))
        re=self.batchnor3_9(self.conv3_9(re))
        node10=F.relu(node9+re)

        re=F.relu(self.batchnor1_10(self.conv1_10(node10)))
        re = F.relu(self.batchnor2_10(self.conv2_10(re)))
        re=self.batchnor3_10(self.conv3_10(re))
        node11=F.relu(node10+re)

        re=F.relu(self.batchnor1_11(self.conv1_11(node11)))
        re = F.relu(self.batchnor2_11(self.conv2_11(re)))
        re=self.batchnor3_11(self.conv3_11(re))
        node12=F.relu(node11+re)

        re=F.relu(self.batchnor1_12(self.conv1_12(node12)))
        re = F.relu(self.batchnor2_12(self.conv2_12(re)))
        re=self.batchnor3_12(self.conv3_12(re))
        node13=F.relu(node12+re)

        re=F.relu(self.batchnor1_13(self.conv1_13(node13)))
        re = F.relu(self.batchnor2_13(self.conv2_13(re)))
        re=self.batchnor3_13(self.conv3_13(re))
        node14=F.relu(node13+re)

        re=F.relu(self.batchnor1_14(self.conv1_14(node14)))
        re = F.relu(self.batchnor2_14(self.conv2_14(re)))
        re=self.batchnor3_14(self.conv3_14(re))
        node14=self.node1_batchnor_14(self.node1_conv_14(node14))
        node15=F.relu(node14+re)

        re = F.relu(self.batchnor1_15(self.conv1_15(node15)))
        re = F.relu(self.batchnor2_15(self.conv2_15(re)))
        re = self.batchnor3_15(self.conv3_15(re))
        node16 = F.relu(node15 + re)

        re = F.relu(self.batchnor1_16(self.conv1_16(node16)))
        re = F.relu(self.batchnor2_16(self.conv2_16(re)))
        re = self.batchnor3_16(self.conv3_16(re))
        node17 =F.avg_pool2d(F.relu(node16 + re),7,1)
        node17 = node17.view(node17.size(0), -1)
        re=self.fc(node17)
        return F.log_softmax(re)