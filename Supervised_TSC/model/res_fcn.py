import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1,64,7,padding=3)
        self.conv2 = nn.Conv1d(64,128,7,padding=3)
        self.block1 = Block()
        self.block2 = Block()
        self.block3 = Block()
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        #x = self.block4(x)
        x = nn.AvgPool1d(96)(x)
        # print(x.shape)
        x = x.view(-1, 128)
        x = x.squeeze()
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.conv1 = nn.Conv1d(128,256,7,padding=3)
        self.conv2 = nn.Conv1d(256,256,5,padding=2)
        self.conv3 = nn.Conv1d(256,128,3,padding=1)
        #self.conv4 = nn.Conv1d(128,128,1)

    def forward(self, x):
        res = x
        x = F.relu(nn.BatchNorm1d(256)(self.conv1(x)))
        x = F.relu(nn.BatchNorm1d(256)(self.conv2(x)))
        x = nn.BatchNorm1d(128)(self.conv3(x))
        #res = self.conv4(res)
        x = x+res*0.2

        return F.relu(x)
