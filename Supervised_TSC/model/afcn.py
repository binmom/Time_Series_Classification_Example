import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input = 96, output = 7):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1,128,9,padding=4)
        self.conv2 = nn.Conv1d(128,256,7,padding=3)
        self.conv3 = nn.Conv1d(256,128,5,padding=2)
        self.pool = nn.AvgPool1d(input)
        self.CA = CALayer(128, 1)
        self.convF = nn.Linear(128,output)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.CA(x)
        x = x.view(-1, 128)
        x = x.squeeze()
        x = self.convF(x)
        
        return x

class CALayer(nn.Module):
    def __init__(self, channel, reduction=10):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        wn = lambda x: torch.nn.utils.weight_norm(x)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                wn(nn.Conv1d(channel, channel // reduction, 1, padding=0, bias=True)),
                nn.ReLU(inplace=True),
                wn(nn.Conv1d(channel // reduction, channel, 1, padding=0, bias=True)),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        # print(x.shape)
        # print(y.shape)
        return x * y
