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

class SSD(nn.Module):
    def __init__(self):
        self.num_class=21
        self.pos_iou=0.5
        self.neg_iou=0.5
        self.pos_neg_rate=3
        self.prior_variances=[0.1,0.1,0.2,0.2]

        super(SSD, self).__init__()
        self.conv1=nn.Conv2d(3,32,3,2,1)
        self.batchnor1=nn.BatchNorm2d(32)

        self.conv2=nn.Conv2d(32,32,3,1,1,groups=32)
        self.batchnor2=nn.BatchNorm2d(32)

        self.conv3=nn.Conv2d(32,64,1)
        self.batchnor3=nn.BatchNorm2d(64)

        self.conv4=nn.Conv2d(64,64,3,2,1,groups=64)
        self.batchnor4=nn.BatchNorm2d(64)

        self.conv5=nn.Conv2d(64,128,1)
        self.batchnor5=nn.BatchNorm2d(128)

        self.conv6=nn.Conv2d(128,128,3,1,1,groups=128)
        self.batchnor6=nn.BatchNorm2d(128)

        self.conv7=nn.Conv2d(128,128,1)
        self.batchnor7=nn.BatchNorm2d(128)

        self.conv8=nn.Conv2d(128,128,3,2,1,groups=128)
        self.batchnor8=nn.BatchNorm2d(128)

        self.conv9=nn.Conv2d(128,256,1)
        self.batchnor9=nn.BatchNorm2d(256)

        self.conv10=nn.Conv2d(256,256,3,1,1,groups=256)
        self.batchnor10=nn.BatchNorm2d(256)

        self.conv11=nn.Conv2d(256,256,1)
        self.batchnor11=nn.BatchNorm2d(256)

        self.conv12=nn.Conv2d(256,256,3,2,1,groups=256)
        self.batchnor12=nn.BatchNorm2d(256)

        self.conv13=nn.Conv2d(256,512,1)
        self.batchnor13=nn.BatchNorm2d(512)

        self.conv14=nn.Conv2d(512,512,3,1,1,groups=512)
        self.batchnor14=nn.BatchNorm2d(512)

        self.conv15=nn.Conv2d(512,512,1)
        self.batchnor15=nn.BatchNorm2d(512)

        self.conv16=nn.Conv2d(512,512,3,1,1,groups=512)
        self.batchnor16=nn.BatchNorm2d(512)

        self.conv17=nn.Conv2d(512,512,1)
        self.batchnor17=nn.BatchNorm2d(512)

        self.conv18=nn.Conv2d(512,512,3,1,1,groups=512)
        self.batchnor18=nn.BatchNorm2d(512)

        self.conv19=nn.Conv2d(512,512,1)
        self.batchnor19=nn.BatchNorm2d(512)

        self.conv20=nn.Conv2d(512,512,3,1,1,groups=512)
        self.batchnor20=nn.BatchNorm2d(512)

        self.conv21=nn.Conv2d(512,512,1)
        self.batchnor21=nn.BatchNorm2d(512)

        self.conv22=nn.Conv2d(512,512,3,1,1,groups=512)
        self.batchnor22=nn.BatchNorm2d(512)

        self.conv23_node1=nn.Conv2d(512,512,1)
        self.batchnor23_node1=nn.BatchNorm2d(512)

        self.conv24=nn.Conv2d(512,512,3,2,1,groups=512)
        self.batchnor24=nn.BatchNorm2d(512)

        self.conv25=nn.Conv2d(512,1024,1)
        self.batchnor25=nn.BatchNorm2d(1024)

        self.conv26=nn.Conv2d(1024,1024,3,1,1,groups=1024)
        self.batchnor26=nn.BatchNorm2d(1024)

        self.conv27_node2=nn.Conv2d(1024,1024,1)
        self.batchnor27_node2=nn.BatchNorm2d(1024)

        self.conv28=nn.Conv2d(1024,256,1)
        self.batchnor28=nn.BatchNorm2d(256)

        self.conv29_node3=nn.Conv2d(256,512,3,2,1)
        self.batchnor29_node3=nn.BatchNorm2d(512)

        self.conv30=nn.Conv2d(512,128,1)
        self.batchnor30=nn.BatchNorm2d(128)

        self.conv31_node4=nn.Conv2d(128,256,3,2,1)
        self.batchnor31_node4=nn.BatchNorm2d(256)

        self.conv32=nn.Conv2d(256,128,1)
        self.batchnor32=nn.BatchNorm2d(128)

        self.conv33_node5=nn.Conv2d(128,256,3,2,1)
        self.batchnor33_node5=nn.BatchNorm2d(256)

        self.conv34=nn.Conv2d(256,64,1)
        self.batchnor34=nn.BatchNorm2d(64)

        self.conv35_node6=nn.Conv2d(64,128,3,2,1)
        self.batchnor35_node6=nn.BatchNorm2d(128)

        self.node1_loc_conv1=nn.Conv2d(512,12,1)
        self.node1_conf_conv1=nn.Conv2d(512,63,1)

        self.node2_loc_conv1=nn.Conv2d(1024,24,1)
        self.node2_conf_conv1=nn.Conv2d(1024,126,1)

        self.node3_loc_conv1=nn.Conv2d(512,24,1)
        self.node3_conf_conv1=nn.Conv2d(512,126,1)

        self.node4_loc_conv1=nn.Conv2d(256,24,1)
        self.node4_conf_conv1=nn.Conv2d(256,126,1)

        self.node5_loc_conv1=nn.Conv2d(256,24,1)
        self.node5_conf_conv1=nn.Conv2d(256,126,1)

        self.node6_loc_conv1=nn.Conv2d(128,24,1)
        self.node6_conf_conv1=nn.Conv2d(128,126,1)

        self.loc=None
        self.conf=None
        self.pri=None


    def forward(self, input):
        re=F.relu(self.batchnor1(self.conv1(input)))
        re=F.relu(self.batchnor2(self.conv2(re)))
        re=F.relu(self.batchnor3(self.conv3(re)))
        re=F.relu(self.batchnor4(self.conv4(re)))
        re=F.relu(self.batchnor5(self.conv5(re)))
        re = F.relu(self.batchnor6(self.conv6(re)))
        re = F.relu(self.batchnor7(self.conv7(re)))
        re = F.relu(self.batchnor8(self.conv8(re)))
        re = F.relu(self.batchnor9(self.conv9(re)))
        re = F.relu(self.batchnor10(self.conv10(re)))
        re = F.relu(self.batchnor11(self.conv11(re)))
        re = F.relu(self.batchnor12(self.conv12(re)))
        re = F.relu(self.batchnor13(self.conv13(re)))
        re = F.relu(self.batchnor14(self.conv14(re)))
        re = F.relu(self.batchnor15(self.conv15(re)))
        re = F.relu(self.batchnor16(self.conv16(re)))
        re = F.relu(self.batchnor17(self.conv17(re)))
        re = F.relu(self.batchnor18(self.conv18(re)))
        re = F.relu(self.batchnor19(self.conv19(re)))
        re = F.relu(self.batchnor20(self.conv20(re)))
        re = F.relu(self.batchnor21(self.conv21(re)))
        re = F.relu(self.batchnor22(self.conv22(re)))
        node1 = F.relu(self.batchnor23_node1(self.conv23_node1(re)))
        re = F.relu(self.batchnor24(self.conv24(node1)))
        re = F.relu(self.batchnor25(self.conv25(re)))
        re = F.relu(self.batchnor26(self.conv26(re)))
        node2 = F.relu(self.batchnor27_node2(self.conv27_node2(re)))
        re = F.relu(self.batchnor28(self.conv28(node2)))
        node3 = F.relu(self.batchnor29_node3(self.conv29_node3(re)))
        re = F.relu(self.batchnor30(self.conv30(node3)))
        node4 = F.relu(self.batchnor31_node4(self.conv31_node4(re)))
        re = F.relu(self.batchnor32(self.conv32(node4)))
        node5 = F.relu(self.batchnor33_node5(self.conv33_node5(re)))
        re = F.relu(self.batchnor34(self.conv34(node5)))
        node6 = F.relu(self.batchnor35_node6(self.conv35_node6(re)))

        node1_loc= self.node1_loc_conv1(node1).permute(0,2,3,1).flatten(1)
        node1_conf = self.node1_conf_conv1(node1).permute(0,2,3,1).flatten(1)

        node2_loc= self.node2_loc_conv1(node2).permute(0,2,3,1).flatten(1)
        node2_conf = self.node2_conf_conv1(node2).permute(0,2,3,1).flatten(1)

        node3_loc= self.node3_loc_conv1(node3).permute(0,2,3,1).flatten(1)
        node3_conf = self.node3_conf_conv1(node3).permute(0,2,3,1).flatten(1)

        node4_loc= self.node4_loc_conv1(node4).permute(0,2,3,1).flatten(1)
        node4_conf = self.node4_conf_conv1(node4).permute(0,2,3,1).flatten(1)

        node5_loc= self.node5_loc_conv1(node5).permute(0,2,3,1).flatten(1)
        node5_conf = self.node5_conf_conv1(node5).permute(0,2,3,1).flatten(1)

        node6_loc= self.node6_loc_conv1(node6).permute(0,2,3,1).flatten(1)
        node6_conf = self.node6_conf_conv1(node6).permute(0,2,3,1).flatten(1)

        loc=torch.cat((node1_loc, node2_loc,node3_loc,node4_loc,node5_loc,node6_loc), 1)
        conf=torch.cat((node1_conf, node2_conf,node3_conf,node4_conf,node5_conf,node6_conf), 1)

        node1_pri=self.priorbox(node1,60,-1,[2])

        node2_pri=self.priorbox(node2, 105, 150, [2,3])

        node3_pri=self.priorbox(node3, 150, 195, [2, 3])

        node4_pri=self.priorbox(node4, 195, 240, [2, 3])

        node5_pri=self.priorbox(node5, 240, 258, [2, 3])

        node6_pri = self.priorbox(node6, 258, 300, [2, 3])

        pri=torch.cat((node1_pri, node2_pri, node3_pri, node4_pri, node5_pri, node6_pri), 2)

        self.conf=conf
        self.pri=pri
        self.loc=loc

        return node6


    def jaccardOverlap_(self,bbox1,bbox2):
        intersect_bbox = self.intersectBBox_(bbox1,bbox2)
        intersect_width=intersect_bbox[2]-intersect_bbox[0]
        intersect_height=intersect_bbox[3]-intersect_bbox[1]

        if intersect_height>0 and intersect_width>0:
            intersect_size=intersect_width*intersect_height
            bbox1_size=self.bboxsize_(bbox1)
            bbox2_size=self.bboxsize_(bbox2)
            return intersect_size/(bbox1_size+bbox2_size-intersect_size)
        else:
            return 0.0
        return

    def intersectBBox_(self,bbox1,bbox2):
        re=torch.zeros(4)
        xmin1 = bbox1[0]
        ymin1 = bbox1[1]
        xmax1 = bbox1[2]
        ymax1 = bbox1[3]

        xmin2 = bbox2[0]
        ymin2 = bbox2[1]
        xmax2 = bbox2[2]
        ymax2 = bbox2[3]

        if xmin2>xmax1 or xmax2<xmin1 or ymin2>ymax1 or ymax2<ymin1:
            pass
        else:
            re[0] = max([xmin1, xmin2])
            re[1] = max([ymin1, ymin2])
            re[2] = min([xmax1, xmax2])
            re[3] = min([ymax1, ymax2])
        return re

    def bboxsize_(self,bbox):
        if bbox[2]<bbox[0] or bbox[3]<bbox[1]:
            return 0
        width=bbox[2]-bbox[0]
        height=bbox[3]-bbox[1]
        return width*height


    def intersectBBox__(self, bbox1, bbox2):
        res = bbox1.clone().detach()

        re0=res[:,0]
        re0[re0<bbox2[0]]=bbox2[0]

        re1=res[:,1]
        re1[re1<bbox2[1]]=bbox2[1]

        re2=res[:,2]
        re2[re2>bbox2[2]]=bbox2[2]

        re3=res[:,3]
        re3[re3>bbox2[3]]=bbox2[3]

        return res

    def bboxsize__(self,bbox):
        res=torch.zeros(bbox.shape[0])
        c=bbox.clone().detach()
        mask=((c[:,2]<c[:,0])+(c[:,3]<c[:,1]))>=1
        mask=mask.view(-1,1)
        mask=mask.expand_as(c)
        c[mask]=0
        return (c[:,2]-c[:,0])*(c[:,3]-c[:,1])

    def jaccardOverlap__(self,prior_boxes,gt_value):
        intersect_bbox=self.intersectBBox__(prior_boxes,gt_value)
        intersect_width=intersect_bbox[:,2]-intersect_bbox[:,0]
        intersect_height=intersect_bbox[:,3]-intersect_bbox[:,1]

        intersect_size=intersect_width*intersect_height

        mask=(intersect_height<=0)
        mask[intersect_width<=0]=1

        bbox1_size=self.bboxsize__(prior_boxes)
        bbox2_size=self.bboxsize_(gt_value)

        res=intersect_size/(bbox1_size+bbox2_size-intersect_size)
        res[mask]=0

        return res

    def creatGlobalIndex(self,prior_boxes,gt_value):
        with torch.no_grad():
            pos_frame = torch.zeros((len(gt_value), (int)(prior_boxes.shape[2] / 4)), dtype=torch.bool)
            pos_gt_value = torch.zeros((len(gt_value), (int)(prior_boxes.shape[2] / 4),5))
            pos_pri_value = torch.zeros((len(gt_value), (int)(prior_boxes.shape[2] / 4), 4))
            neg_frame = torch.zeros((len(gt_value), (int)(prior_boxes.shape[2] / 4)), dtype=torch.bool)

            pri = prior_boxes[0, 0, :].view(-1, 4)
            for i in range(len(gt_value)):
                max_iou=torch.ones(pri.shape[0])*-1
                max_id=torch.zeros(pri.shape[0],5)
                min_iou=torch.ones(pri.shape[0])*100

                for n in range(gt_value[i].shape[0]):
                    iou=self.jaccardOverlap__(prior_boxes=pri,gt_value=gt_value[i][n,0:4])
                    mask=max_iou<iou
                    max_iou[mask]=iou[mask]
                    mask_id=mask.unsqueeze(mask.dim()).expand_as(max_id)
                    sum=mask.sum()
                    if sum>0:
                        max_id[mask_id]=gt_value[i][n,:].repeat(sum)
                    mask=min_iou>iou
                    min_iou[mask]=iou[mask]
                mask=max_iou>self.pos_iou
                pos_frame[i,:]=mask
                neg_frame[i,:]=min_iou<self.neg_iou
                neg_frame[i,:][mask]=0

                mask_id=mask.unsqueeze(mask.dim()).expand_as(max_id)
                pos_gt_value[i,:][mask_id]=max_id[mask_id]

                mask_pri = mask.unsqueeze(mask.dim()).expand_as(pri)
                pos_pri_value[i, :][mask_pri] = pri[mask_pri]

        # for i in range(distri_frame.shape[0]):
        #     pos_indic = []
        #     for j in range(distri_frame.shape[1]):
        #         max_iou=-1
        #         max_id=0
        #         neg=False
        #         for n in range(gt_value[i].shape[0]):
        #             iou=self.jaccardOverlap_(gt_value[i][n,0:4],prior_boxes[0,0,4*j:4*j+4])
        #             if iou>self.pos_iou:
        #                 max_iou=max(iou, max_iou)
        #                 max_id=n
        #             if iou<self.neg_iou:
        #                 neg=True
        #         if max_iou>-1:
        #             distri_frame[i,j]=max_id
        #             pos_count=pos_count+1
        #             pos_indic.append(j)
        #         elif neg:
        #             distri_frame[i,j]=-2
        #             neg_all_count=neg_all_count+1
        #     pos_indics.append(pos_indic)
        # neg_cout=min(neg_all_count,pos_count*self.pos_neg_rate)
        # distri_frame[distri_frame>0]=1
        # distri_frame[distri_frame<0]=0
        # ssd=distri_frame.sum()

        return pos_frame,pos_gt_value,pos_pri_value,neg_frame


    def encodeBBox_(self, prior_bbox, bbox):
        res = torch.zeros(4)
        prior_width = prior_bbox[2] - prior_bbox[0]
        prior_height = prior_bbox[3] - prior_bbox[1]
        prior_center_x = (prior_bbox[0] + prior_bbox[2]) / 2
        prior_center_y = (prior_bbox[1] + prior_bbox[3]) / 2

        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        bbox_center_y = (bbox[1] + bbox[3]) / 2

        res[0] = (bbox_center_x - prior_center_x) / prior_width /self.prior_variances[0]
        res[1] = (bbox_center_y - prior_center_y) / prior_height /self.prior_variances[1]
        res[2] = math.log(bbox_width / prior_width) /self.prior_variances[2]
        res[3] = math.log(bbox_height / prior_height) /self.prior_variances[3]

        return res

    def encodeBBox__(self, prior_bbox, bbox):
        res = torch.zeros(bbox.shape)
        prior_width = prior_bbox[:,2] - prior_bbox[:,0]
        prior_height = prior_bbox[:,3] - prior_bbox[:,1]
        prior_center_x = (prior_bbox[:,0] + prior_bbox[0,2]) / 2
        prior_center_y = (prior_bbox[:,1] + prior_bbox[:,3]) / 2

        bbox_width = bbox[:,2] - bbox[:,0]
        bbox_height = bbox[:,3] - bbox[:,1]
        bbox_center_x = (bbox[:,0] + bbox[:,2]) / 2
        bbox_center_y = (bbox[:,1] + bbox[:,3]) / 2

        res[:,0] = (bbox_center_x - prior_center_x) / prior_width /self.prior_variances[0]
        res[:,1] = (bbox_center_y - prior_center_y) / prior_height /self.prior_variances[1]
        res[:,2] = ((bbox_width / prior_width) /self.prior_variances[2]).log()
        res[:,3] = ((bbox_height / prior_height) /self.prior_variances[3]).log()

        return res

    def decodeBBox__(self, prior_bbox, encode):
        res = torch.zeros(encode.shape)
        prior_width = prior_bbox[:,2] - prior_bbox[:,0]
        prior_height = prior_bbox[:,3] - prior_bbox[:,1]
        prior_center_x = (prior_bbox[:,0] + prior_bbox[0,2]) / 2
        prior_center_y = (prior_bbox[:,1] + prior_bbox[:,3]) / 2

        bbox_center_x=encode[:,0]*self.prior_variances[0]*prior_width+prior_center_x
        bbox_center_y=encode[:,1]*self.prior_variances[1]*prior_height+prior_center_y
        bbox_width=encode[:,2].exp()*self.prior_variances[2]*prior_width
        bbox_height=encode[:,3].exp()*self.prior_variances[3]*prior_height


        res[:,0]=(2*bbox_center_x-bbox_width)/2
        res[:,1]=(2*bbox_center_x+bbox_width)/2
        res[:,2]=(2*bbox_center_y-bbox_height)/2
        res[:,3]=(2*bbox_center_y+bbox_height)/2

        return res

    def creatPositiveSample(self,pos_frame,pos_gt_value,pos_pri_value,loc,conf):
        loc = loc.view(loc.shape[0], -1, 4)
        conf = conf.view(conf.shape[0], -1, self.num_class)
        with torch.no_grad():
            pos_frame_pri=pos_frame.unsqueeze(pos_frame.dim()).expand_as(pos_pri_value)
            pos_frame_gt = pos_frame.unsqueeze(pos_frame.dim()).expand_as(pos_gt_value)
            pos_frame_conf=pos_frame.unsqueeze(pos_frame.dim()).expand_as(conf)

            pri=pos_pri_value[pos_frame_pri].view(-1,4)
            gt=pos_gt_value[pos_frame_gt].view(-1,5)

        pos_loc_target=self.encodeBBox__(prior_bbox=pri,bbox=gt[:,0:4])
        pos_loc_pre=loc[pos_frame_pri].view(-1,4)

        pos_conf_target=gt[:,4]+1
        pos_conf_pre=conf[pos_frame_conf].view(-1,self.num_class)

        # pos_loc_target=torch.zeros(pos_count*4)
        # pos_conf_target=torch.zeros(pos_count*self.num_class)
        # pos_loc_pre=torch.zeros(pos_count*4)
        # pos_conf_pre=torch.zeros(pos_count*self.num_class)
        #
        # count=0
        # for i in range(len(pos_indics)):
        #     for j in range(len(pos_indics[i])):
        #         pos_idx = pos_indics[i][j]
        #         gt_idx=distri_frame[i,pos_idx]
        #
        #         # create loc
        #         for n in range(4):
        #             pos_loc_pre[4*count+n]=loc[i,4*pos_idx+n]
        #         target=self.encodeBBox_(prior_bbox=prior_boxes[0, 0, 4*pos_idx:4 * pos_idx + 4],bbox=gt_value[i][gt_idx, 0:4])
        #         for n in range(4):
        #             pos_loc_target[4*count+n]=target[n]
        #
        #         # create conf
        #         for n in range(self.num_class):
        #             pos_conf_pre[self.num_class*count+n]=conf[i,self.num_class*pos_idx+n]
        #         label=(int)(gt_value[i][gt_idx,4]+1)  #background is 0
        #         for n in range(self.num_class):
        #             pos_conf_target[self.num_class*count+n]=1 if(n==label) else 0
        #         count=count+1
        #pos_loc_target, pos_conf_target, pos_loc_pre, pos_conf_pre=0
        return pos_loc_target,pos_conf_target,pos_loc_pre,pos_conf_pre

    def creatNegativeSample(self,neg_frame,conf,pos_num):
        # code1: evaluate score


        neg_nums=pos_num*self.pos_neg_rate

        conf=conf.view(conf.shape[0],-1,self.num_class)
        neg_frame_conf=neg_frame.unsqueeze(neg_frame.dim()).expand_as(conf)
        neg=conf[neg_frame_conf].view(-1,self.num_class)
        scores=neg.softmax(0)[:,0]

        sort=scores.sort(0, descending=True)

        neg_conf_pre=neg[sort.indices[0:neg_nums],:]
        neg_conf_target=torch.zeros(neg_conf_pre.shape[0])


        '''
        scores=torch.zeros(neg_all_count,3)
        neg_conf_target=torch.zeros(self.num_class*neg_cout)
        neg_conf_pre=torch.zeros(self.num_class*neg_cout)
        neg_indics=[]
        count=0
        for i in range(distri_frame.shape[0]):
            neg_indics.append([])
            for j in range(distri_frame.shape[1]):
                if distri_frame[i,j]<-1:#indicate it is neg_sample
                    sin_conf=conf[i,self.num_class*j:self.num_class*j+self.num_class].softmax(0)[0]
                    scores[count,0]=sin_conf
                    scores[count,1]=i
                    scores[count,2]=j
                    count=count+1
        scores=scores.sort(0,descending=True).values
        count=0
        for i in range(neg_cout):
            batch_id=(int)(scores[i,1])
            loc_id=(int)(scores[i,2])
            for j in range(self.num_class):
                neg_conf_pre[self.num_class*count+j]=conf[batch_id,self.num_class*loc_id+j]
            neg_indics[batch_id].append(loc_id)
            count=count+1
        '''
        return neg_conf_target,neg_conf_pre

    def nms(self,boxes, scores, overlap=0.5, top_k=200):
        keep = scores.new(scores.size(0)).zero_().long()
        if boxes.numel() == 0:
            return keep
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = torch.mul(x2 - x1, y2 - y1)
        v, idx = scores.sort(0)  # sort in ascending order
        # I = I[v >= 0.01]
        idx = idx[-top_k:]  # indices of the top-k largest vals
        xx1 = boxes.new()
        yy1 = boxes.new()
        xx2 = boxes.new()
        yy2 = boxes.new()
        w = boxes.new()
        h = boxes.new()

        # keep = torch.Tensor()
        count = 0
        while idx.numel() > 0:
            i = idx[-1]  # index of current largest val
            # keep.append(i)
            keep[count] = i
            count += 1
            if idx.size(0) == 1:
                break
            idx = idx[:-1]  # remove kept element from view
            # load bboxes of next highest vals
            torch.index_select(x1, 0, idx, out=xx1)
            torch.index_select(y1, 0, idx, out=yy1)
            torch.index_select(x2, 0, idx, out=xx2)
            torch.index_select(y2, 0, idx, out=yy2)
            # store element-wise max with next highest score
            xx1 = torch.clamp(xx1, min=x1[i])
            yy1 = torch.clamp(yy1, min=y1[i])
            xx2 = torch.clamp(xx2, max=x2[i])
            yy2 = torch.clamp(yy2, max=y2[i])
            w.resize_as_(xx2)
            h.resize_as_(yy2)
            w = xx2 - xx1
            h = yy2 - yy1
            # check sizes of xx1 and xx2.. after each iteration
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)
            inter = w * h
            # IoU = i / (area(a) + area(b) - i)
            rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
            union = (rem_areas - inter) + area[i]
            IoU = inter / union  # store result in iou
            # keep only elements with an IoU <= overlap
            idx = idx[IoU.le(overlap)]
        return keep, count

    def test(self,data_loader,mode=True):

        # num_images = len(data_loader)
        # for i in range(num_images):
        #     print('Testing image {:d}/{:d}....'.format(i + 1, num_images))
        #     img = data_loader.pull_image(i)
        #     img_id, annotation = data_loader.pull_anno(i)
        #     # x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        #     # x = Variable(x.unsqueeze(0))

        batch_iterator = iter(data_loader)
        images, targets = next(batch_iterator)
        for i in range(300):
            self.forward(images)
            pri = self.pri.data[0,0,:].view(-1, 4)
            loc = self.loc.data
            conf = self.conf.data

            loc = loc.view(loc.shape[0], -1, 4)
            conf = conf.view(conf.shape[0], -1, self.num_class)
            conf=F.softmax(conf,dim=-1)

            num = loc.size(0)  # batch size
            num_priors = pri.size(0)
            output = torch.zeros(num, self.num_class, 200, 5)
            conf_preds = conf.transpose(2, 1)

        # Decode predictions into bboxes.
            for i in range(num):
                decoded_boxes = self.decodeBBox__(pri,loc[i])
                # For each class, perform nms
                conf_scores = conf_preds[i].clone()

                for cl in range(1, self.num_class):
                    c_mask = conf_scores[cl].gt(0.01)
                    scores = conf_scores[cl][c_mask]
                    if scores.size(0) == 0:
                        continue
                    l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                    boxes = decoded_boxes[l_mask].view(-1, 4)
                    # idx of highest scoring and non-overlapping boxes per class
                    ids, count = self.nms(boxes, scores)
                    output[i, cl, :count] = \
                        torch.cat((scores[ids[:count]].unsqueeze(1),
                                boxes[ids[:count]]), 1)
            flt = output.contiguous().view(num, -1, 5)
            _, idx = flt[:, :, 0].sort(1, descending=True)
            _, rank = idx.sort(1)
            flt[(rank < 200).unsqueeze(-1).expand_as(flt)].fill_(0)
            return output
        return

    def train(self,data_loader,mode=True):
        # data.DataLoader
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9,
                              weight_decay=0.0005)
        batch_iterator = iter(data_loader)
        images, targets = next(batch_iterator)
        for i in range(120000):
            self.forward(images.cuda())
            pri = self.pri.data
            loc = self.loc
            conf = self.conf

            #print(targets)

            pos_frame,pos_gt_value,pos_pri_value,neg_frame=self.creatGlobalIndex(prior_boxes=pri,gt_value=targets)
            pos_loc_target,pos_conf_target,pos_loc_pre,pos_conf_pre=self.creatPositiveSample(pos_frame=pos_frame,pos_gt_value=pos_gt_value,pos_pri_value=pos_pri_value,loc=loc,conf=conf)
            pos_num = pos_conf_target.shape[0]
            neg_conf_target,neg_conf_pre=self.creatNegativeSample(neg_frame=neg_frame,conf=conf,pos_num=pos_num)




            # conf_target=torch.zeros(pos_conf_target.shape[0]+neg_conf_target.shape[0])
            # conf_pre=torch.zeros(pos_conf_pre.shape[0]+neg_conf_pre.shape[0])
            # loc_target=torch.zeros(pos_loc_target.shape)
            # loc_pre=torch.zeros(pos_loc_pre.shape)


            conf_target=torch.cat([pos_conf_target,neg_conf_target],0).long()
            conf_pre=torch.cat([pos_conf_pre, neg_conf_pre], 0)
            # loc_target=torch.cat([pos_loc_target],0,out=.data)
            # loc_pre=torch.cat([pos_loc_pre],0,out=.data)

            optimizer.zero_grad()
            loss_l = F.smooth_l1_loss(pos_loc_pre, pos_loc_target.cuda(), size_average=False)
            loss_c = F.cross_entropy(conf_pre, conf_target.long().cuda(), size_average=False)

            N=pos_num
            loss_c/=N
            loss_l/=N

            loss=loss_l+loss_c
            loss.backward()
            try:
                images, targets = next(batch_iterator)
            except StopIteration:
                batch_iterator=iter(data_loader)
                images,targets=next(batch_iterator)
            if i%20==0:
                print("loss:",loss.item(),'iterator:',i)
            optimizer.step()


            if (i%999)==0:
                torch.save(self.state_dict(), "./tmp/"+str(i)+"_ssd_par.pth")

            # loss = F.nll_loss(F.log_softmax(conf_pre), conf_target) + F.smooth_l1_loss(loc_pre,loc_target)
            # loss.backward()
            #
            # loc_pred_data_grad = torch.zeros(loc.shape)
            # conf_pred_data_grad = torch.zeros(conf.shape)
            '''
            self.reset(all_match_indices_=all_match_indices_, neg_indices=neg_indices,
                       loc_pred_data_grad=loc_pred_data_grad, conf_pred_data_grad=conf_pred_data_grad,
                       loc_pred_data_ori_grad=loc_pred_data.grad, conf_pred_data_ori_grad=conf_pred_data.grad)

            asd = loc_pred_data_grad[loc_pred_data_grad != 0]
            asd = conf_pred_data_grad[conf_pred_data_grad != 0]

            self.loc.backward(loc_pred_data_grad, retain_graph=True)
            self.conf.backward(conf_pred_data_grad)
            '''
            '''
            all_gt_bboxes=self.getGroundTruth(gt_data=targets)
            prior_bboxes,prior_variances=self.getPriorBBoxes(pri)
            all_loc_preds=self.getLocPredictions(loc)
            all_match_overlaps,all_match_indices_=self.findMatches(all_loc_preds=all_loc_preds,all_gt_bboxes=all_gt_bboxes,prior_bboxes=prior_bboxes,prior_variances=prior_variances,params=None)
            neg_indices,num_poses,num_neges=self.getNegsamples(all_match_overlaps,conf,all_gt_bboxes,all_match_indices_)

            loc_pred_data = torch.zeros(1,num_poses*4,requires_grad=True);
            loc_gt_data = torch.zeros(1,num_poses*4);
            self.encodeLocPrediction(all_loc_preds=all_loc_preds,all_gt_bboxes=all_gt_bboxes,all_match_indices=all_match_indices_,prior_bboxes=prior_bboxes,prior_variances=prior_variances,loc_pred_data=loc_pred_data.data,loc_gt_data=loc_gt_data.data)

            conf_pred_data = torch.zeros(num_poses+num_neges,self.num_class,requires_grad=True);
            conf_gt_data = torch.zeros(num_poses+num_neges);
            self.encodeConfPrediction(conf_data=conf,all_match_indices=all_match_indices_,all_neg_indices=neg_indices,all_gt_bboxes=all_gt_bboxes,conf_pred_data=conf_pred_data.data,conf_gt_data=conf_gt_data.data)

            loss=F.nll_loss(F.log_softmax(conf_pred_data),conf_gt_data.long())+F.smooth_l1_loss(loc_pred_data,loc_gt_data)
            loss.backward()

            loc_pred_data_grad=torch.zeros(loc.shape)
            conf_pred_data_grad=torch.zeros(conf.shape)

            self.reset(all_match_indices_=all_match_indices_,neg_indices=neg_indices,loc_pred_data_grad=loc_pred_data_grad,conf_pred_data_grad=conf_pred_data_grad,loc_pred_data_ori_grad=loc_pred_data.grad,conf_pred_data_ori_grad=conf_pred_data.grad)

            asd=loc_pred_data_grad[loc_pred_data_grad!=0]
            asd=conf_pred_data_grad[conf_pred_data_grad!=0]

            self.loc.backward(loc_pred_data_grad,retain_graph=True)
            self.conf.backward(conf_pred_data_grad)
            '''
        return

    def reset(self,all_match_indices_,neg_indices,loc_pred_data_grad,conf_pred_data_grad,loc_pred_data_ori_grad,conf_pred_data_ori_grad):
        count=0
        for i in range(len(all_match_indices_)):
            try:
                list(all_match_indices_[i].keys()).index(-1)
            except:
                continue
            match_indices=all_match_indices_[i][-1]
            for j in range(match_indices.shape[1]):
                if match_indices[0,j]>-1:
                    for n in range(4):
                        loc_pred_data_grad[i,4*j+n]=loc_pred_data_ori_grad[0,count*4+n]
                    for n in range(self.num_class):
                        conf_pred_data_grad[i,self.num_class*j+n]=conf_pred_data_ori_grad[count,n]
                    count=count+1

            for j in range(len(neg_indices[i])):
                idx=neg_indices[i][j]
                for n in range(self.num_class):
                    conf_pred_data_grad[i,self.num_class*idx+n]=conf_pred_data_ori_grad[count,n]
                count=count+1
        return

    def input_size(self):
        return [300,300]

    def encodeConfPrediction(self,conf_data,all_match_indices,all_neg_indices,all_gt_bboxes,conf_pred_data,conf_gt_data):
        count = 0
        for i in range(conf_data.shape[0]):
            try:
                list(all_gt_bboxes.keys()).index(i)
            except:
                continue
            match_indices=all_match_indices[i]
            try:
                list(match_indices.keys()).index(-1)
            except:
                continue

            match_index=match_indices[-1]

            for j in range(match_index.shape[1]):
                if match_index[0,j]<=-1:
                    continue
                gt_label=all_gt_bboxes[i][match_index[0,j]][0,0]
                idx=count
                conf_gt_data[idx]=gt_label
                for k in range(self.num_class): #这个位置要改
                    conf_pred_data[count,k]=conf_data[i,j*self.num_class+k]
                #conf_pred_data[count,]=conf_data[i,j*self.num_class:(j+1)*self.num_class]
                count = count + 1

            for n in range(len(all_neg_indices[i])):
                j = all_neg_indices[i][n]
                for k in range(self.num_class): #这个位置要改
                    conf_pred_data[count,k]=conf_data[i,j*self.num_class+k]
                conf_gt_data[count]=0
                count=count+1
        return

    def encodeLocPrediction(self,all_loc_preds,all_gt_bboxes,all_match_indices,prior_bboxes,prior_variances,loc_pred_data,loc_gt_data):
        count = 0
        for i in range(len(all_loc_preds)):
            try:
                list(all_match_indices[i].keys()).index(-1)
            except:
                continue
            match_index=all_match_indices[i][-1]
            loc_pred=all_loc_preds[i][-1]
            for j in range(match_index.shape[1]):
                if match_index[0,j]<=-1:
                    continue

                gt_idx = match_index[0,j]
                gt_bbox=all_gt_bboxes[i][gt_idx]
                gt_encode=self.encodeBBox(prior_bboxes[j],prior_variances[j],gt_bbox)
                loc_gt_data[0,count*4]=gt_encode[0,1]
                loc_gt_data[0, count * 4+1]=gt_encode[0,2]
                loc_gt_data[0, count * 4+2]=gt_encode[0,3]
                loc_gt_data[0, count * 4+3]=gt_encode[0,4]

                loc_pred_data[0,count*4]=loc_pred[j][0,1]
                loc_pred_data[0, count * 4+1]=loc_pred[j][0,2]
                loc_pred_data[0, count * 4+2]=loc_pred[j][0,3]
                loc_pred_data[0, count * 4+3]=loc_pred[j][0,4]
                count=count+1

        return

    def encodeBBox(self,prior_bbox,prior_variances,bbox):
        res=torch.zeros(1,8)
        prior_width=prior_bbox[0,3]-prior_bbox[0,1]
        prior_height=prior_bbox[0,4]-prior_bbox[0,2]
        prior_center_x=(prior_bbox[0,1]+prior_bbox[0,3])/2
        prior_center_y=(prior_bbox[0,2]+prior_bbox[0,4])/2

        bbox_width=bbox[0,3]-bbox[0,1]
        bbox_height=bbox[0,4]-bbox[0,2]
        bbox_center_x=(bbox[0,1]+bbox[0,3])/2
        bbox_center_y=(bbox[0,2]+bbox[0,4])/2

        res[0,1]=(bbox_center_x-prior_center_x)/prior_width/prior_variances[0]
        res[0,2]=(bbox_center_y-prior_center_y)/prior_height/prior_variances[1]
        res[0,3]=math.log(bbox_width/prior_width)/prior_variances[2]
        res[0,4]=math.log(bbox_height/prior_height)/prior_variances[3]

        return res

    def getNegsamples(self,all_match_overlaps,conf,all_gt_bboxes,all_match_indices_):
        neg_indices=[]
        all_scores = self.softmax(conf,all_gt_bboxes,all_match_indices_)
        num_poses=0
        num_neges=0
        for i in range(len(all_match_overlaps)):
            try:
                list(all_match_overlaps[i].keys()).index(-1)
            except:
                neg_indices.append([])
                continue
            match_indices = all_match_indices_[i][-1]
            num_pos=match_indices[match_indices>-1].shape[0]
            num_poses=num_poses+num_pos
            num_neg=num_pos*3
            overlaps=all_match_overlaps[i][-1]
            scores=all_scores[i]
            neg=[]
            neg_indic=[]
            for j in range(overlaps.shape[1]):
                if overlaps[0,j]<0.5:
                    neg.append([scores[j],j])
            self.quickSort(neg,0,len(neg)-1)
            num_neg=min(num_neg,len(neg))
            num_neges=num_neges+num_neg
            for j in range(num_neg):
                neg_indic.append(neg[j][1])
            neg_indices.append(neg_indic)
        return neg_indices,num_poses,num_neges

    def partition(self,arr, low, high):
        i = (low - 1)
        pivot = arr[high][0]

        for j in range(low, high):
            if arr[j][0] > pivot:
                i = i + 1
                arr[i][0], arr[j][0] = arr[j][0], arr[i][0]
                arr[i][1],arr[j][1]=arr[j][1],arr[i][1]

        arr[i + 1][0], arr[high][0] = arr[high][0], arr[i + 1][0]
        arr[i + 1][1], arr[high][1] = arr[high][1], arr[i + 1][1]
        return (i + 1)


    def quickSort(self,arr, low, high):
        if low < high:
            pi =self.partition(arr, low, high)

            self.quickSort(arr, low, pi - 1)
            self.quickSort(arr, pi + 1, high)

    def softmax(self,conf,all_gt_bboxes,all_match_indices_):
        re=[]
        for i in range(len(all_match_indices_)):
            indices=all_match_indices_[i]
            small_re=[]
            for j in range(int(conf.shape[1]/self.num_class)):
                startindex=self.num_class*j
                label=0

                try:
                    list(indices.keys()).index(-1)
                    indices = indices[-1]
                    if indices[0, j] > -1:
                        label = all_gt_bboxes[i][indices[0, j]][0, 0]
                except:
                    pass

                maxval=conf[i,0]
                sum=0
                for k in range(self.num_class):
                    maxval=max(conf[i,startindex+k],maxval)
                for k in range(self.num_class):
                    sum+=math.exp(conf[i,startindex+k]-maxval)
                prob=math.exp(conf[i,startindex+label]-maxval)/sum
                small_re.append(-math.log(max(prob,1.175494351e-38)))
            re.append(small_re)
        return re

    def getLocPredictions(self,loc_data):
        re=[]
        for i in range(loc_data.shape[0]):
            dic={}
            re.append(dic)
            for p in range(int(loc_data.shape[1]/4)):
                bbox=torch.zeros(1,8)
                bbox[0, 1] = loc_data[i,4*p]
                bbox[0, 2] = loc_data[i, 4*p+1]
                bbox[0, 3] = loc_data[i,4*p+2]
                bbox[0, 4] = loc_data[i,4*p+3]
                dic.setdefault(-1, []).append(bbox)
        return re

    def findMatches(self,all_loc_preds,all_gt_bboxes,prior_bboxes,prior_variances,params):
        all_match_overlaps=[]
        all_match_indices=[]
        for i in range(len(all_loc_preds)):
            match_indices={}
            match_overlaps={}
            if all_gt_bboxes.get(i)==None:
                all_match_indices.append(match_indices)
                all_match_overlaps.append(match_overlaps)
                continue
            gt_bboxes=all_gt_bboxes[i]
            temp_match_indices,temp_match_overlaps= self.matchBBox(gt_bboxes=gt_bboxes,pred_bboxes=prior_bboxes,label=-1,overlap_threshold=0.5)
            match_indices.setdefault(-1,temp_match_indices)
            match_overlaps.setdefault(-1,temp_match_overlaps)

            all_match_indices.append(match_indices)
            all_match_overlaps.append(match_overlaps)
        return all_match_overlaps,all_match_indices

    def matchBBox(self,gt_bboxes,pred_bboxes,label,overlap_threshold):
        match_indices=-torch.ones(1,len(pred_bboxes)).int()
        match_overlaps=torch.zeros(1,len(pred_bboxes)).float()
        overlaps={}
        gt_indices=[]
        for i in range(len(gt_bboxes)):
            gt_indices.append(i)
        for i in range(len(pred_bboxes)):
            for j in range(len(gt_bboxes)):
                overlap = self.jaccardOverlap(pred_bboxes[i],gt_bboxes[gt_indices[j]])
                if overlap>1e-6:
                    match_overlaps[0,i]=max([match_overlaps[0,i],overlap])
                    overlaps.setdefault(i,{}).setdefault(j,overlap)

        gt_pool=[]
        for i in range(len(gt_bboxes)):
            gt_pool.append(i)
        while len(gt_pool)>0:
            max_overlap = -1
            max_idx = -1
            max_gt_idx = -1
            for it in overlaps.keys():
                if match_indices[0,i]!=-1:
                    continue
                for p in range(len(gt_pool)):
                    j=gt_pool[p]
                    #print(overlaps[it].get(p))
                    try:
                        list(overlaps[it].keys()).index(j)
                    except:
                        continue
                    #if overlaps[it].get(p)==None:
                    #    continue
                    if overlaps[it][j]>max_overlap:
                        max_idx=it
                        max_gt_idx=j
                        max_overlap=overlaps[it][j]
            if max_idx==-1:
                break
            else:
                match_indices[0,max_idx]=gt_indices[max_gt_idx]
                match_overlaps[0,max_idx]=max_overlap
                gt_pool.remove(max_gt_idx)
        for it in overlaps.keys():
            i=it
            if match_indices[0,i]!=-1:
                continue
            max_gt_idx=-1
            max_overlap=-1.0
            for j in range(len(gt_bboxes)):
                try:
                    list(overlaps[it].keys()).index(j)
                except:
                    continue
                overlap=overlaps[it][j]
                if overlap>=overlap_threshold and overlap>max_overlap:
                    max_gt_idx=j
                    max_overlap=overlap
            if max_gt_idx!=-1:
                match_indices[0,i]=gt_indices[max_gt_idx]
                match_overlaps[0,i]=max_overlap
        return match_indices,match_overlaps

    def jaccardOverlap(self,bbox1,bbox2):
        intersect_bbox = self.intersectBBox(bbox1,bbox2)
        intersect_width=intersect_bbox[0,3]-intersect_bbox[0,1]
        intersect_height=intersect_bbox[0,4]-intersect_bbox[0,2]

        if intersect_height>0 and intersect_width>0:
            intersect_size=intersect_width*intersect_height
            bbox1_size=self.bboxsize(bbox1)
            bbox2_size=self.bboxsize(bbox2)
            return intersect_size/(bbox1_size+bbox2_size-intersect_size)
        else:
            return 0.0
        return

    def intersectBBox(self,bbox1,bbox2):
        re=torch.zeros(1,8)
        xmin1 = bbox1[0, 1]
        ymin1 = bbox1[0, 2]
        xmax1 = bbox1[0, 3]
        ymax1 = bbox1[0, 4]

        xmin2 = bbox2[0, 1]
        ymin2 = bbox2[0, 2]
        xmax2 = bbox2[0, 3]
        ymax2 = bbox2[0, 4]

        if xmin2>xmax1 or xmax2<xmin1 or ymin2>ymax1 or ymax2<ymin1:
            re[0, 1]=0
            re[0, 2]=0
            re[0, 3]=0
            re[0, 4]=0
        else:
            re[0, 1] = max([xmin1, xmin2])
            re[0, 2] = max([ymin1, ymin2])
            re[0, 3] = min([xmax1, xmax2])
            re[0, 4] = min([ymax1, ymax2])
        return re

    def getPriorBBoxes(self,prior_data):
        prior_bboxes=[]
        prior_variances=[]
        for i in range(int(prior_data.shape[2]/4)):
            start_idx=i*4
            bbox=torch.zeros(1,8)
            bbox[0, 1]=prior_data[0,0,start_idx]         #xmin
            bbox[0, 2] = prior_data[0, 0, start_idx+1]   #ymin
            bbox[0, 3] = prior_data[0, 0, start_idx+2]   #xmax
            bbox[0, 4] = prior_data[0, 0, start_idx+3]   #ymax
            bbox[0, 6] = self.bboxsize(bbox)
            prior_bboxes.append(bbox)
            prior_variances.append([prior_data[0,1,start_idx],prior_data[0,1,start_idx+1],prior_data[0,1,start_idx+2],prior_data[0,1,start_idx+3]])
        return prior_bboxes,prior_variances

    def getGroundTruth(self,gt_data):
        re={}
        for i in range(len(gt_data)):
            item_id=i
            for j in range(gt_data[i].shape[0]):
                bbox = torch.zeros(1, 8)
                bbox[0,0]=gt_data[i][j,4]+1
                bbox[0, 1] = gt_data[i][j,0]+1
                bbox[0, 2] = gt_data[i][j,1]+1
                bbox[0, 3] = gt_data[i][j,2]+1
                bbox[0, 4] = gt_data[i][j,3]+1
                bbox[0, 6] = self.bboxsize(bbox)
                re.setdefault(item_id,[]).append(bbox)

        # re={}
        # for i in range(gt_data.shape[2]):
        #     start_idx=0
        #     item_id=int(gt_data[0,0,i,start_idx].item())
        #     if item_id==-1:
        #         continue
        #     label=gt_data[0,0,i,start_idx+1]
        #     difficult=(True if(gt_data[0,0,i,start_idx+7]>=1) else False)
        #     bbox=torch.zeros(1,8)
        #     bbox[0,0]=label
        #     bbox[0, 1] = gt_data[0,0,i,start_idx+3]
        #     bbox[0, 2] = gt_data[0,0,i,start_idx+4]
        #     bbox[0, 3] = gt_data[0,0,i,start_idx+5]
        #     bbox[0, 4] = gt_data[0,0,i,start_idx+6]
        #     bbox[0, 5] = gt_data[0,0,i,start_idx+7]
        #     bbox[0, 6] = self.bboxsize(bbox)
        #     re.setdefault(item_id,[]).append(bbox)
        return re

    def bboxsize(self,bbox):
        if bbox[0,3]<bbox[0,1] or bbox[0,4]<bbox[0,2]:
            return 0
        width=bbox[0,3]-bbox[0,1]
        height=bbox[0,4]-bbox[0,2]
        return width*height

    def priorbox(self,input,min_size,max_size,aspect_ratio):
        aspect_ratio_=[]
        aspect_ratio_.append(1.0)
        variance_=[0.1,0.1,0.2,0.2]
        already_exist=False
        for i in range(len(aspect_ratio)):
            ar=aspect_ratio[i]
            for j in range(len(aspect_ratio_)):
                if math.fabs(ar-aspect_ratio_[j])<1e-6:
                    already_exist=True
                    break
            if not already_exist:
                aspect_ratio_.append(ar)
                aspect_ratio_.append(1.0/ar)

        num_priors_=len(aspect_ratio_)
        if max_size>0:
            num_priors_+=1

        idx = 0
        layer_width=input.shape[2]
        layer_height=input.shape[3]

        img_width=300
        img_height=300

        step_w=img_width/layer_width
        step_h=img_height/layer_height


        dim=layer_height * layer_width * num_priors_ * 4;
        re=torch.zeros(1,2,dim)
        pribox_index=0
        for h in range(layer_height):
            for w in range(layer_width):
                center_x=(w+0.5)*step_w
                center_y=(h+0.5)*step_h
                if min_size>0:
                    box_width=box_height=min_size
                    re[0,pribox_index,idx]=(center_x-box_width/2.0)/img_width
                    idx+=1
                    re[0,pribox_index,idx]=(center_y-box_height/2.0)/img_height
                    idx += 1
                    re[0,pribox_index,idx]=(center_x+box_width/2.0)/img_width
                    idx += 1
                    re[0,pribox_index,idx]=(center_y+box_height/2.0)/img_height
                    idx += 1

                    if max_size>0:
                        box_width=box_height=math.sqrt(min_size*max_size)
                        re[0,pribox_index,idx]=(center_x - box_width / 2.0) / img_width
                        idx += 1
                        re[0,pribox_index,idx]=(center_y - box_height / 2.0) / img_height
                        idx += 1
                        re[0,pribox_index,idx]=(center_x + box_width / 2.0) / img_width
                        idx += 1
                        re[0,pribox_index,idx]=(center_y + box_height / 2.0) / img_height
                        idx += 1

                    for r in range(len(aspect_ratio_)):
                        ar=aspect_ratio_[r]
                        if math.fabs(ar-1.0)<1e-6:
                            continue
                        box_width=min_size*math.sqrt(ar)
                        box_height=min_size/math.sqrt(ar)
                        re[0,pribox_index,idx]=(center_x - box_width / 2.0) / img_width
                        idx += 1
                        re[0,pribox_index,idx]=(center_y - box_height / 2.0) / img_height
                        idx += 1
                        re[0,pribox_index,idx]=(center_x + box_width / 2.0) / img_width
                        idx += 1
                        re[0,pribox_index,idx]=(center_y + box_height / 2.0) / img_height
                        idx += 1
        idx=0
        for h in range(layer_height):
            for w in range(layer_width):
                for i in range(num_priors_):
                    for j in range(len(variance_)):
                        re[0,1,idx]=variance_[j]
                        idx+=1


        return re


#loc_data   16*7668
#conf_data  16*40257
#prior_data 1*2*7668
#gt_data    1*1*37*8
def train():
    ds = voc.VOCDetection('E:\VOC027\VOCdevkit\VOCdevkit',
                          transform=voc.SSDAugmentation())
    num_images = len(ds)
    data_loader = data.DataLoader(ds, 16, num_workers=4, shuffle=True, collate_fn=voc.detection_collate)

    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    net=SSD()
    net.cuda()
    #input=torch.rand(3,3,300,300)
    net.train(data_loader=data_loader)
#make_dot(net(input)).view()
#flops, params = profile(net, inputs=(input, ))

#torch.save(net, '\model.pkl')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    summary(net, input_size=(3, 300, 300))
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))


def test():
    net=SSD()
    net.load_state_dict(torch.load('E:\VOC027\ssd_par.pth',map_location='cpu'))

    ds = voc.VOCDetection('E:\VOC027\VOCdevkit',
                          [('2007', 'test')],transform=voc.BaseTransform(300,(104, 117, 123)))
    num_images = len(ds)
    data_loader = data.DataLoader(ds, 16, num_workers=4, shuffle=True, collate_fn=voc.detection_collate)
    #net.cuda()

    net.test(data_loader=data_loader)
