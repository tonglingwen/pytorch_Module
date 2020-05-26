import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch
import data.ilsvrc2012 as ilsvrc2012
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class Depthwise_Convolutional(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1,bias=False):
        super(Depthwise_Convolutional,self).__init__()
        self.conv=nn.Sequential(
                                Regular_Convolutional(in_channels,in_channels,3,stride=stride,groups=in_channels,padding=1,bias=bias),
                                Regular_Convolutional(in_channels, out_channels,1,stride=1,bias=bias)
                            )


    def forward(self, input):
        res=self.conv(input)
        return res

class Regular_Convolutional(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=False, padding_mode='zeros'):
        super(Regular_Convolutional,self).__init__()
        self.conv=nn.Sequential(
                                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups,
                                        bias=bias, padding_mode=padding_mode),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(),
                            )


    def forward(self, input):
        res=self.conv(input)
        return res

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet,self).__init__()
        self.conv0=Regular_Convolutional(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.depthconv1=Depthwise_Convolutional(in_channels=32,out_channels=64)
        self.depthconv2 = Depthwise_Convolutional(in_channels=64, out_channels=128,stride=2)
        self.depthconv3=Depthwise_Convolutional(in_channels=128,out_channels=128)
        self.depthconv4=Depthwise_Convolutional(in_channels=128,out_channels=256,stride=2)
        self.depthconv5=Depthwise_Convolutional(in_channels=256,out_channels=256)
        self.depthconv6=Depthwise_Convolutional(in_channels=256,out_channels=512,stride=2)
        self.depthconv7=nn.Sequential(
            Depthwise_Convolutional(in_channels=512, out_channels=512),
            Depthwise_Convolutional(in_channels=512, out_channels=512),
            Depthwise_Convolutional(in_channels=512, out_channels=512),
            Depthwise_Convolutional(in_channels=512, out_channels=512),
            Depthwise_Convolutional(in_channels=512, out_channels=512)
        )
        self.depthconv8=Depthwise_Convolutional(in_channels=512,out_channels=1024,stride=2)
        self.depthconv9=Depthwise_Convolutional(in_channels=1024,out_channels=1024)

        #self.fc=nn.Linear(1024,1000)


    def forward(self, input):
        res=self.conv0(input)
        res=self.depthconv1(res)
        res = self.depthconv2(res)
        res = self.depthconv3(res)
        res = self.depthconv4(res)
        res = self.depthconv5(res)
        res = self.depthconv6(res)
        node1 = self.depthconv7(res)
        #self.depthconv7在SSD中为node1(第一个分支)


        node2 = self.depthconv8(node1)
        node2 = self.depthconv9(node2)
        #self.depthconv9在SSD中为node2(第二个分支)

        # size=res.size()
        # res=F.avg_pool2d(res,size[2],1)
        # res=res.view(-1,size[1])
        # res=F.softmax(self.fc(res),dim=-1)
        return node1,node2

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def imagetest2():
    net=MobileNet()
    net.load_state_dict(torch.load(r'E:\Work\pytorch\pytorch_Module\Module\my_mobile.pth'))
    net.eval()
    net=net.cuda()

    datasets_il =ilsvrc2012.ILSVRC2012DatasetValidation(r"F:\ml-dataset\ilsvrc2012\unzip")

    val_loader = torch.utils.data.DataLoader(datasets_il,
        batch_size=32, shuffle=True,
        num_workers=4)

    batch_iterator = iter(val_loader)
    images, targets = next(batch_iterator)

    for i in range(16):
        img2 = transforms.ToPILImage()(images[i, :, :, :])
        img2.save(str(i) + ".jpg")

    prec1_s = 0
    prec5_s=0
    n=0
    for i in range(1000):
        with torch.no_grad():
            res=net(images.cuda())
            prec1, prec5 = accuracy(res.data, targets.cuda(), topk=(1, 5))
            images, targets = next(batch_iterator)
            print("prec1=",prec1.item(),"   prec5=",prec5.item())
            prec1_s=prec1_s+prec1
            prec5_s=prec5_s+prec5
            n=i
            if n==100:
                break
    print("prec1_all=",prec1_s.item()/n,"   prec5_all=",prec5_s.item()/n)


def imagetest():
    net=MobileNet()
    net.load_state_dict(torch.load(r'E:\Work\pytorch\pytorch_Module\Module\my_mobile.pth'))
    net.eval()
    net=net.cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(r"F:\ml-dataset\ilsvrc2012\unzip\ILSVRC2012_img_val", transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=32, shuffle=True,
        num_workers=4)

    b = [0 for _ in range(50000)]
    # batch_iterator = iter(val_loader)
    # images, targets = next(batch_iterator)
    indes = val_loader.batch_sampler.sampler.data_source.imgs
    for i in range(len(indes)):
        nds=indes[i][0].split('\\')[6]
        nds=nds.split('_')[2]
        nds=int(nds.split('.')[0])-1
        b[nds]=indes[i][1]

    filename = 'test_text.txt'
    with open(filename, 'w') as file_object:
        for i in range(len(b)):
            file_object.write(str(b[i])+"\n")

    #prec1_s=0
    # prec5_s=0
    # n=0
    # for i, (input, target) in enumerate(val_loader):
    #     with torch.no_grad():
    #         res=net(input.cuda())
    #         prec1, prec5 = accuracy(res.data, target.cuda(), topk=(1, 5))
    #         print("prec1=",prec1.item(),"   prec5=",prec5.item())
    #         prec1_s=prec1_s+prec1
    #         prec5_s=prec5_s+prec5
    #         n=i
    #         if n==100:
    #             break
    # print("prec1_all=",prec1_s.item()/n,"   prec5_all=",prec5_s.item()/n)
        #images, targets = next(batch_iterator)

# net=MobileNet()
# mk=torch.load('my_mobile.pth')
# net.load_state_dict(mk,strict=False)
# mk=net.state_dict()
# print('da')

# mobile=torch.load("mobilenet.pth")
# mk=net.state_dict()
# #
# i=0
# mok=list(mobile.keys())
# mkk=list(mk.keys())
# for k in range(len(mobile)):
#     #print((mok[k]))
#     if mobile[mok[k]].size()!=mk[mkk[k]].size():
#         print('Error')
#     else:
#         mk[mkk[k]]=mobile[mok[k]]
#
# torch.save(mk, "my_mobile.pth")
# print(i)

#net=net.cuda()

#torch.save(net, 'model.pkl')
#torch.save(net.state_dict(), "_ssd_par.pth")

#summary(net, input_size=(3, 300, 300),device='cpu')
