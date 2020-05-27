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
import Module.MobileNet as mobile


class SSD_mobilenet(nn.Module):
    def __init__(self):
        self.num_class=21
        self.pos_iou=0.5
        self.neg_iou=0.5
        self.pos_neg_rate=3
        self.prior_variances=[0.1,0.1,0.2,0.2]

        super(SSD_mobilenet, self).__init__()

        self.basenet=mobile.MobileNet()

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
        node1,node2=self.basenet(input)

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

    def bboxsize_(self,bbox):
        if bbox[2]<bbox[0] or bbox[3]<bbox[1]:
            return 0
        width=bbox[2]-bbox[0]
        height=bbox[3]-bbox[1]
        return width*height

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

        return pos_frame,pos_gt_value,pos_pri_value,neg_frame

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

        return pos_loc_target,pos_conf_target,pos_loc_pre,pos_conf_pre
    #
    def creatNegativeSample(self,neg_frame,conf,pos_num):
        neg_nums=pos_num*self.pos_neg_rate

        conf=conf.view(conf.shape[0],-1,self.num_class)
        neg_frame_conf=neg_frame.unsqueeze(neg_frame.dim()).expand_as(conf)
        neg=conf[neg_frame_conf].view(-1,self.num_class)
        scores=neg.softmax(0)[:,0]

        sort=scores.sort(0, descending=True)

        neg_conf_pre=neg[sort.indices[0:neg_nums],:]
        neg_conf_target=torch.zeros(neg_conf_pre.shape[0])

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

    def detection(self):
        pri = self.pri.data[0, 0, :].view(-1, 4)
        loc = self.loc.data
        conf = self.conf.data

        loc = loc.view(loc.shape[0], -1, 4)
        conf = conf.view(conf.shape[0], -1, self.num_class)
        conf = F.softmax(conf, dim=-1)

        num = loc.size(0)  # batch size
        num_priors = pri.size(0)
        output = torch.zeros(num, self.num_class, 200, 5)
        conf_preds = conf.transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = self.decodeBBox__(pri, loc[i])
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
        img2 = transforms.ToPILImage()(images[0])
        img2.show()
        for i in range(300):
            self.forward(images)
            detections=self.detection()
            for n in range(detections.size(1)):
                j = 0
                while detections[0, n, j, 0] >= 0.6:
                    pt = (detections[0, n, j, 1:]*300).cpu().numpy()
                    score = detections[0, n, j, 0]
                    print(pt[0],pt[1],pt[2],pt[3],score)
                    j+=1
            return
        return

    def train(self,data_loader,mode=True):
        # data.DataLoader



        self.basenet.load_state_dict(torch.load(r'/voc/my_mobile.pth'),strict=False)



        for name, value in self.basenet.named_parameters():
            value.requires_grad = False

        #sp=self.parameters()

        params = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = optim.SGD(params, lr=0.001, momentum=0.9,weight_decay=0.0005)
        batch_iterator = iter(data_loader)
        images, targets = next(batch_iterator)
        # for i in range(16):
        #     img2 = transforms.ToPILImage()(images[i,:,:,:])
        #     img2.save(str(i)+".jpg")

        for i in range(120000*20):
            mk = self.basenet.state_dict()
            mk1 = self.state_dict()
            self.forward(images.cuda())
            pri = self.pri.data
            loc = self.loc
            conf = self.conf

            pos_frame,pos_gt_value,pos_pri_value,neg_frame=self.creatGlobalIndex(prior_boxes=pri,gt_value=targets)
            pos_loc_target,pos_conf_target,pos_loc_pre,pos_conf_pre=self.creatPositiveSample(pos_frame=pos_frame,pos_gt_value=pos_gt_value,pos_pri_value=pos_pri_value,loc=loc,conf=conf)
            pos_num = pos_conf_target.shape[0]
            neg_conf_target,neg_conf_pre=self.creatNegativeSample(neg_frame=neg_frame,conf=conf,pos_num=pos_num)


            conf_target=torch.cat([pos_conf_target,neg_conf_target],0).long()
            conf_pre=torch.cat([pos_conf_pre, neg_conf_pre], 0)

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

        return

    def input_size(self):
        return [300,300]

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

# net=SSD_mobilenet()
#
#
# summary(net, input_size=(3, 300, 300),device='cpu')


def train():
    ds = voc.VOCDetection(r'/voc/VOCdevkit',
                          transform=voc.SSDAugmentation())
    data_loader = data.DataLoader(ds, 16, num_workers=1, shuffle=True, collate_fn=voc.detection_collate)


    net=SSD_mobilenet()
    #net.load_state_dict(torch.load(r'/app/pytorch_Module/tmp/0_ssd_par.pth'))
    net.cuda()
    net.train(data_loader=data_loader)



def test():
    net=SSD_mobilenet()
    net.load_state_dict(torch.load('F:\\voc\ssd_par.pth',map_location='cpu'))

    ds = voc.VOCDetection('F:\\voc\VOCtest_06-Nov-2007\VOCdevkit',
                          [('2007', 'test')],transform=voc.BaseTransform(300,(104, 117, 123)))
    num_images = len(ds)
    data_loader = data.DataLoader(ds, 16, num_workers=4, shuffle=True, collate_fn=voc.detection_collate)
    #net.cuda()

    net.test(data_loader=data_loader)

