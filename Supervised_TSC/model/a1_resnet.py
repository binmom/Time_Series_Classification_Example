import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input = 96, output = 7):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1,32,5,padding=2)
        self.conv2 = nn.Conv1d(32,64,5,padding=2)
        self.block1 = Block()
        self.block2 = Block()
        self.block3 = Block()
        self.pool = nn.AvgPool1d(input)
        self.fc1 = nn.Linear(64,32)
        self.fc2 = nn.Linear(32,output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x =self.pool(x)
        # print(x.shape)
        x = x.view(-1, 64)
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
        self.CA = CALayer(64,8)
        self.conv4 = nn.Conv1d(64,64,1)

    def forward(self, x):
        res = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.CA(x)
        res = self.conv4(res)
        x = x+res

        return F.relu(x)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.conv_du = nn.Sequential(
            wn(nn.Conv1d(channel, channel//reduction, 1, padding=0, bias=True)),
            nn.ReLU(inplace=True),
            wn(nn.Conv1d(channel//reduction, channel, 1 ,padding=0, bias=True)),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
