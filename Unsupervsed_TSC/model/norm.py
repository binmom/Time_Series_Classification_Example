import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math

def global_max_pooling(x):
    dim = x.size()[1]
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
            nn.Conv1d(1, 16, 7, padding=3),
            nn.ReLU(True),
            nn.AvgPool1d(2),
            nn.Conv1d(16, 64, 5, padding=2),
            nn.ReLU(True),
            nn.AvgPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(True)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose1d(128,64,3,stride=2,padding=1),
            nn.ReLU(True),
            nn.Conv1d(64,64,5,padding=2),
            nn.ReLU(True)
        )

        exp = math.floor(math.log(input/9, 4))
        # print(exp)
        att = []
        if exp != 0:
            for i in range(exp):
                att.append(nn.ConvTranspose1d(64,64,4,stride=4,padding=0))
                att.append(nn.ReLU(True))
                att.append(nn.Conv1d(64,64,5,padding=2))
        att.append(nn.Conv1d(64,4,5,padding=2))
        self.decoder2 = nn.Sequential(*att)
        leng = 4**exp
        self.leng = leng*9
        self.fc1 = nn.Linear(4*self.leng,input)
        self.fc2 = nn.Linear(input,input)
        self.input = input

    def forward(self, x):
        # print(x.size())
        x = self.encoder(x)
        latent, indices = global_max_pooling(x)

        norm = (latent*latent).sum()
        norm = torch.tensor([[norm]]*128).cuda()
        latent = latent/torch.sqrt(norm)
        latent = latent * 0.01
        # print((latent*latent).sum())

        # print(latent.size())
        latent_5 = nn.MaxUnpool1d(kernel_size=5)(latent, indices)
        # print(latent.size())
        x = self.decoder1(latent_5)
        # print(x.size())
        x = self.decoder2(x)
        # print(x.size())
        x = x.view(-1, 4*self.leng)
        # print(x.size())
        x = self.fc1(x)
        x = self.fc2(x)
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
