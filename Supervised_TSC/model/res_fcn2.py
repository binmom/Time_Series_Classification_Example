import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1,128,7,padding=3)
        self.conv2 = nn.Conv1d(128,256,5,padding=2)
        self.conv3 = nn.Conv1d(256,128,3,padding=1)
        self.conv4 = nn.Conv1d(128,256,5,padding=2)
        self.conv5 = nn.Conv1d(256,128,3,padding=1)
        self.conv6 = nn.Conv1d(128,256,5,padding=2)
        self.conv7 = nn.Conv1d(256,128,3,padding=1)        
        self.convF = nn.Linear(128,7)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        s = x
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = s+x
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))                
        x = F.avg_pool1d(x,96)
        x = x.view(-1,128)
        x = x.squeeze()
        x = self.convF(x)
        
        return x
