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

class SSD(nn.Module):
    def __init__(self):
        self.num_class=21
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

    def train(self, mode=True):
        pri=self.pri.data
        loc=self.loc.data
        conf=self.conf.data

        gd=torch.rand(1,1,34,8)

        all_gt_bboxes=self.getGroundTruth(gt_data=gd)
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
        for i in range(gt_data.shape[2]):
            start_idx=0
            item_id=int(gt_data[0,0,i,start_idx].item())
            if item_id==-1:
                continue
            label=gt_data[0,0,i,start_idx+1]
            difficult=(True if(gt_data[0,0,i,start_idx+7]>=1) else False)
            bbox=torch.zeros(1,8)
            bbox[0,0]=label
            bbox[0, 1] = gt_data[0,0,i,start_idx+3]
            bbox[0, 2] = gt_data[0,0,i,start_idx+4]
            bbox[0, 3] = gt_data[0,0,i,start_idx+5]
            bbox[0, 4] = gt_data[0,0,i,start_idx+6]
            bbox[0, 5] = gt_data[0,0,i,start_idx+7]
            bbox[0, 6] = self.bboxsize(bbox)
            re.setdefault(item_id,[]).append(bbox)
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

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
net=SSD()
input=torch.rand(1,3,300,300)
net(input)
net.train()
#make_dot(net(input)).view()
#flops, params = profile(net, inputs=(input, ))

#torch.save(net, '\model.pkl')
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
summary(net, input_size=(3, 300, 300))
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))