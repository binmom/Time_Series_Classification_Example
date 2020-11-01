from sklearn.decomposition import PCA
import sklearn.decomposition as dcp
import torch
from data_load import train_load, test_load
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda')

model_name = 'naive'
dataset = 'ECGFiveDays'

net_G = torch.load('pth/'+ model_name +'/' + dataset + '_G' + '.pth')
net_D = torch.load('pth/'+ model_name +'/' + dataset + '_D' + '.pth')

time, train = train_load(dataset, batch=1)

embed = np.zeros([1,40])


with torch.no_grad():
    for _, check in enumerate(train):
        input = check.to(device)
        output, latent = net_G(input)
        # print(latent.cpu().numpy().shape)
        embed = np.vstack([embed, latent.cpu().numpy()])

embed = embed[1:,:]
# print(embed.shape)

print('PCA Analysis start')

# pca1 = PCA(n_components=3)
pca1 = dcp.KernelPCA(n_components=3, kernel='rbf', gamma=0.0433, fit_inverse_transform=True)

X_low = pca1.fit_transform(embed)
I_X = pca1.inverse_transform(X_low)

# print(pca1.explained_variance_ratio_)
# print(X_low.shape)

plt.figure(figsize= (10, 10))
plt.scatter(X_low[:,0], X_low[:,2])
plt.savefig('pca.png')
plt.show()

