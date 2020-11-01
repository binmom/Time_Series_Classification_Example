import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input = 96, output = 7):
        super(Net, self).__init__()
        self.TA = TALayer(input,1)
        self.conv1 = nn.Conv1d(1,128,7,padding=3)
        self.conv2 = nn.Conv1d(128,256,5,padding=2)
        self.conv3 = nn.Conv1d(256,128,3,padding=1)
        self.pool = nn.AvgPool1d(input)
        self.CA = CALayer(128, 1)
        self.convF = nn.Linear(128,output)
        

    def forward(self, x):
        x = x.permute(0,2,1)
        # print(x.shape)
        x = self.TA(x)
        x = x.permute(0,2,1)
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

class TALayer(nn.Module):
    def __init__(self, channel, reduction=10):
        super(TALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        wn = lambda x: torch.nn.utils.weight_norm(x)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                wn(nn.Conv1d(channel, channel // reduction, 5, padding=2, bias=True)),
                nn.ReLU(inplace=True),
                wn(nn.Conv1d(channel // reduction, channel, 5, padding=2, bias=True)),
                nn.Sigmoid()
        )

    def forward(self, x):
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        z = torch.cat([y1,y2],2)
        z = self.conv_du(z)
        z = self.avg_pool(z)
        return x * z
