import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1,32,5,padding=2)
        self.conv2 = nn.Conv1d(32,64,5,padding=2)
        self.block1 = Block()
        self.block2 = Block()
        self.block3 = Block()
        self.block4 = Block()
        self.block5 = Block()
        self.block6 = Block()
        self.block7 = Block()
        self.block8 = Block()


        self.fc1 = nn.Linear(12*64,64)
        self.fc2 = nn.Linear(64,7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = nn.AvgPool1d(8)(x)
        # print(x.shape)
        x = x.view(-1, 12*64)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.squeeze()

        return x

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.conv1 = nn.Conv1d(64,64,7,padding=3)
        self.conv2 = nn.Conv1d(64,64,5,padding=2)
        self.conv3 = nn.Conv1d(64,64,3,padding=1)
        self.conv4 = nn.Conv1d(64,64,1)

    def forward(self, x):
        res = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        #res = self.conv4(res)
        x = x+res*0.2

        return F.relu(x)


