import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

channel = 32
grow = 32
Layer = 6
num_block = 20

class RDB_conv(nn.Module):
	def __init__(self, channel, grow):
		super(RDB_conv,self).__init__()
		self.conv = nn.Sequential(*[
			nn.Conv1d(channel,grow,3,padding=1)
		])
	def forward(self, x):
		out = F.relu(self.conv(x))
		return torch.cat((x, out),1)

class RDB(nn.Module):
	def __init__(self):
		super(RDB,self).__init__()

		convs = []
		for i in range(Layer):
			convs.append(RDB_conv(channel + grow*i, grow))
		self.convs = nn.Sequential(*convs)
		self.LFF = nn.Conv1d(channel+grow*Layer, channel, 1)
	
	def forward(self, x):
		return self.LFF(self.convs(x)) + x

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv1d(1,channel,7,padding=3)
		self.conv2 = nn.Conv1d(channel,channel,5,padding=2)
		self.RDBs = nn.ModuleList()
		for i in range(num_block):
			self.RDBs.append(RDB())
		self.GFF = nn.Conv1d(num_block*channel, channel, 1)
		self.fc1 = nn.Linear(channel*96, 96)
		self.fc2 = nn.Linear(96,7)

	def forward(self, x):
		s = self.conv1(x)
		x = self.conv2(s)

		RDBs_out = []
		for i in range(num_block):
			x = self.RDBs[i](x)
			RDBs_out.append(x)

		x = self.GFF(torch.cat(RDBs_out,1))
		x += s
		x = x.view(-1, channel*96)
		x = self.fc1(x)
		x = self.fc2(x)
		x = x.squeeze()

		return x
