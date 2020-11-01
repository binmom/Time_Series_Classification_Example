import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input = 96, output = 7):
        super(Net, self).__init__()
        self.Attention = Attention(time = input, out_channel = 64)
        self.block1 = Block(ca = input)
        self.block2 = Block(ca = input)
        self.block3 = Block(ca = input)
        self.expool = nn.Conv1d(input,1,1)
        self.convF = nn.Linear(64,output)

    def forward(self, x):
        x = self.Attention(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.permute(0,2,1)
        x = self.expool(x)
        x = x.permute(0,2,1)
        x = x.view(-1,64)
        x = x.squeeze()
        x = self.convF(x)
        out = torch.sigmoid(x)

        return out

class Block(nn.Module):
    def __init__(self, ca = 96):
        super(Block, self).__init__()
        self.conv1 = nn.Conv1d(64,64,7,padding=3)
        self.conv2 = nn.Conv1d(64,64,5,padding=2)
        self.conv3 = nn.Conv1d(64,64,3,padding=1)
        self.conv4 = nn.Conv1d(64,64,1)
        self.CA = CALayer(64,8,time=ca)

    def forward(self, x):
        res = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        res = self.conv4(res)
        x = F.relu(x+res)
        x = self.CA(x)

        return x

class Attention(nn.Module):
	def __init__(self, time=96, out_channel=64):
		super(Attention, self).__init__()
		wn = lambda x: torch.nn.utils.weight_norm(x)
		self.conv_du = nn.Sequential(
			wn(nn.Conv1d(time, time//4, 7, padding=3, bias=True)),
			nn.Sigmoid(),
			wn(nn.Conv1d(time//4, time, 3 ,padding=1, bias=True)),
			nn.Sigmoid()
		)
		self.expansion = nn.Conv1d(1,out_channel,7,padding=3)

	def forward(self, x):
		x = x.permute(0,2,1)
		z = self.conv_du(x)
		x = x * z
		x = x.permute(0,2,1)
		x = self.expansion(x)
		return x

class CALayer(nn.Module):
	def __init__(self, channel, reduction=8, time = 96):
		super(CALayer, self).__init__()
		self.extract = nn.Conv1d(time, 1, 1)
		wn = lambda x: torch.nn.utils.weight_norm(x)
		self.conv_du = nn.Sequential(
			wn(nn.Conv1d(channel, channel//reduction, 1, padding=0, bias=True)),
			nn.ReLU(inplace=True),
			wn(nn.Conv1d(channel//reduction, channel, 1 ,padding=0, bias=True)),
			nn.Sigmoid()
		)

	def forward(self, x):
		y = x.permute(0,2,1)
		y = self.extract(y)
		y = y.permute(0,2,1)
		y = self.conv_du(y)
		return x * y
