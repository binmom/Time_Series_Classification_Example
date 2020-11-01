import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from data_load import train_load, test_load
# from data_load2 import train_load, test_load
# from data_load_class import train_load, test_load, real_test_load
from model import *

def main(filename = 'ElectricDevices', batch = 100, epoch = 1000):

    cls, time, train, train_label, val, val_label = train_load(filename ,batch= batch)

    device = torch.device('cuda')

    # print('Data Loaded')

    # net = basic_cnn.Net()
    net = sunday.Net(input = time, output = cls).to(device)
    # net = mnist.Net().to(device)
    # criterion = F.nll_loss()
    # criterion = nn.Focal

    # net = torch.load('resnet.pth')

    # optimizer = optim.SGD(net.parameters(), lr = 1e-2, momentum=(0.5))
    optimizer = optim.Adam(net.parameters(), lr = 1e-3)

    ## training

    # print('Training Start')

    acc_Best = 0
    pth_name = 'pth/sunday/' + filename + '.pth'
    torch.save(net, pth_name)
    for epo in range(epoch):
        for i in range(len(train)):
            optimizer.zero_grad()
            input = train[i].to(device)
            label = train_label[i]
            output = net(input).cpu()
            # print(output[0].shape)
            loss = F.cross_entropy(output, label)
            # loss = nn.CrossEntropyLoss(output, label)
            # pt = torch.exp(-loss)
            # F_loss = (1-pt)*loss
            # loss.backward()
            loss.backward()
            optimizer.step()
        # print('batch ', batch, ' epoch :  ', epo+1)

        O = 0
        X = 0

        with torch.no_grad():
            for i in range(len(val)):
                input = val[i].to(device)
                label = val_label[i]
                output = net(input).cpu()
                # print(torch.max(output,0))
                _, index = torch.max(output,0)
                # print(_, index)
                index = index.numpy()
                # print(index,label)
                label = label[0]
                if label == index:
                    O +=1
                else:
                    X +=1
                # else:
            #     print(label, index)

        acc = O/(O+X)
        if acc_Best < acc:
            acc_Best = acc
            pth_name = 'pth/sunday/'+filename+'.pth'
            torch.save(net, pth_name)
        # print('Best_acc : ', acc_Best)
    print('val : ', acc_Best)
    O = 0
    X = 0
    test, test_label = test_load(filename = filename)

    device = torch.device('cuda')

    model = torch.load(pth_name)

    with torch.no_grad():
        for i in range(len(test)):
            input = test[i].to(device)
            label = test_label[i]
            output = model(input).cpu()
            # print(torch.max(output,0))
            _, index = torch.max(output, 0)
            # print(_, index)
            index = index.numpy()
            # print(index,label)
            label = label[0]
            if label == index:
                O += 1
            else:
                X += 1
            # else:
        #     print(label, index)

    acc = O / (O + X)
    print('testing : ', acc)
    return acc

