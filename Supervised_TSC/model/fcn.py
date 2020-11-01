import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input = 96, output = 7):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1,128,7,padding=3)
        self.conv2 = nn.Conv1d(128,256,5,padding=2)
        self.conv3 = nn.Conv1d(256,128,3,padding=1)
        self.pool =nn.AvgPool1d(input)
        self.convF = nn.Linear(128, output)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128)
        x = x.squeeze()
        x = self.convF(x)
        
        return x
