import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
from Module import ResNet34
from Module import TestNet


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch_Module')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--module', default='testnet', help='module')
    args = parser.parse_args()

    return args

def transform(img):
    img=img.resize((28, 28), Image.ANTIALIAS)
    re=transforms.ToTensor()(img)
    return re

#net=ResNet34.ResNet34(1,10)


batch_size = 64
test_batch_size = 32







def test(net,test_loader,cuda=False):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda :
            data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda())
        else:
            data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        # sum up batch loss
        #.data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: , Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train(net,train_loader,epoch,cuda=False):
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            if cuda:
                data=data.cuda()
                target=target.cuda()
            out = net(data)
            loss = F.nll_loss(out, target)
            net.zero_grad()     # zeroes the gradient buffers of all parameters
            optimizer.zero_grad()
            print(ep,":",loss)
            loss.backward()
            optimizer.step()

#device = torch.device('cuda:0')
#net.to(device)


#net.load_state_dict(torch.load('\parameter.pkl'))
#net = torch.load('\model.pkl')

#torch.save(net, '\model.pkl')
#torch.save(net.state_dict(), '\parameter.pkl')

def main(args):
    net = False
    if args.module == 'testnet':
        net = TestNet.TestNet(1, 10)
    if args.module == 'resnet34':
        net = ResNet34.ResNet34(1, 10)

    if not net:
        print("choose a module please!")
        return 0
    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transform)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False)
    if args.device == 'gpu':
        net.cuda()
    train(net,train_loader,1, args.device == 'gpu')
    test(net,train_loader,args.device == 'gpu')

if __name__ == "__main__":
    args = parse_args()
    main(args)