import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input = 96, output = 7):
        super(Net, self).__init__()
        self.Attention = Attention(time = input, out_channel = 128)
        self.conv2 = nn.Conv1d(128,256,7,padding=3)
        self.conv3 = nn.Conv1d(256,128,5,padding=2)
        self.export1 = nn.Conv1d(input,input,1)
        self.export2 = nn.Conv1d(input,1,1)
        self.CA = CALayer(128, 1)
        self.fc1 = nn.Linear(128,128)
        self.fc2 = nn.Linear(128,output)
        

    def forward(self, x):
        x = self.Attention(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.permute(0,2,1)
        x = self.export1(x)
        x = self.export2(x)
        x = x.permute(0,2,1)
        x = self.CA(x)
        x = x.view(-1, 128)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.squeeze()
        
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

class Attention(nn.Module):
    def __init__(self, time=96, out_channel=128):
        super(Attention, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.conv_du = nn.Sequential(
            wn(nn.Conv1d(time, time*2, 7, padding=3, bias=True)),
            nn.ReLU(),
            wn(nn.Conv1d(time*2, time*2, 5, padding=2, bias=True)),
            nn.ReLU(),
            wn(nn.Conv1d(time*2, time, 3 ,padding=1, bias=True)),
            nn.Sigmoid()
        )
        self.expansion = nn.Sequential(
            nn.Conv1d(1,out_channel,7,padding=3),
            nn.Conv1d(out_channel,out_channel,5,padding=2),
            nn.Conv1d(out_channel,out_channel,3,padding=1)
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        z = self.conv_du(x)
        x = x * z
        x = x.permute(0,2,1)
        x = self.expansion(x)
        return F.relu(x)
