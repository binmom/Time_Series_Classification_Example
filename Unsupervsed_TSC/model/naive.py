import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math

def global_max_pooling(x):
    dim = x.size()[2]
    # print(x.size())
    ret, _ = nn.MaxPool1d(dim, return_indices=True)(x)
    return ret, _

def global_avg_pooling(x):
    dim = x.size()[1]
    ret = nn.AvgPool1d(dim)(x)
    return ret

class Net(nn.Module):
    def __init__(self, input=96):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 5, 9, padding=4),
            nn.ReLU(True),
            nn.AvgPool1d(2),
            nn.Conv1d(5, 10, 7, padding=3),
            nn.ReLU(True),
            nn.AvgPool1d(2),
            nn.Conv1d(10, 20, 5, padding=2),
            nn.ReLU(True),
            nn.AvgPool1d(2),
            nn.Conv1d(20, 40, 3, padding=1),
            nn.ReLU(True)
        )

        self.fc1 = nn.Linear(40, 60)
        self.fc2 = nn.Linear(60, input)
        self.fc3 = nn.Linear(input,input)
        self.input = input

    def forward(self, x):
        # print(x.size())
        x = self.encoder(x)
        latent, indices = global_max_pooling(x)
        # print(x.size())
        latent = latent.view(-1, 40)
        # print(x.size())
        x = self.fc1(latent)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        x = x.squeeze()
        x = x.view(-1,1,self.input)
        return x, latent

class Discriminator(nn.Module):
    def __init__(self, input=96):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(1,128,7,padding=3)
        self.conv2 = nn.Conv1d(128,128,5,padding=2)
        self.conv3 = nn.Conv1d(128,128,3,padding=1)
        self.pool =nn.AvgPool1d(input)
        self.convF = nn.Linear(128, 1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128)
        x = x.squeeze()
        x = self.convF(x)
        x = torch.sigmoid(x)
        return x


