import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input = 96, output = 7):
        super(Net, self).__init__()
        self.Attention = Attention(time = input, out_channel = 128)
        self.conv2 = nn.Conv1d(128,256,5,padding=2)
        self.CA1 = CALayer(256,8,time=input)
        self.conv3 = nn.Conv1d(256,128,3,padding=1)
        self.CA2 = CALayer(128,8,time=input)
        self.expool = nn.Conv1d(input,1,1)
        self.convF = nn.Linear(128, output)
        

    def forward(self, x):
        x = self.Attention(x)
        x = F.relu(self.conv2(x))
        x = self.CA1(x)
        x = F.relu(self.conv3(x))
        x = self.CA2(x)
        x = x.permute(0,2,1)
        x = self.expool(x)
        x = x.permute(0,2,1)
        x = x.view(-1, 128)
        x = x.squeeze()
        x = self.convF(x)
        
        return x

class Attention(nn.Module):
	def __init__(self, time=96, out_channel=128):
		super(Attention, self).__init__()
		wn = lambda x: torch.nn.utils.weight_norm(x)
		self.conv_du = nn.Sequential(
			wn(nn.Conv1d(time, time//4, 7, padding=3, bias=True)),
			nn.Sigmoid(),
			wn(nn.Conv1d(time//4, time, 3 ,padding=1, bias=True)),
			nn.Sigmoid()
		)
		self.expansion = nn.Conv1d(1,128,7,padding=3)

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
