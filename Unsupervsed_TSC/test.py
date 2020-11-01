import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from data_load import train_load, test_load
from model import *
import sklearn.metrics as metrics
import matplotlib
import matplotlib.pyplot as plt
import statistics as stats

matplotlib.use("template")

print('Testing...')

device = torch.device('cuda')

model_name = 'naive'
dataset = 'ECGFiveDays'
# dataset = 'TwoLeadECG'
# dataset = 'GunPointAgeSpan'
# dataset = 'Herring'
dataset = 'ItalyPowerDemand'
# dataset = 'MoteStrain'
# dataset = 'ToeSegmentation2'
# dataset = 'Wafer'


net_G = torch.load('pth/'+ model_name +'/' + dataset + '_G' + '.pth')
net_D = torch.load('pth/'+ model_name +'/' + dataset + '_D' + '.pth')

time, test, test_label = test_load(dataset)
print('Data Loaded')

tp = 0
tn = 0
fp = 0
fn = 0

reg_loss = nn.MSELoss().to(device)

prob = []
dis = []
true = []

with torch.no_grad():
    for i in range(len(test)):
        input = test[i].to(device)
        label = test_label[i]
        output, latent = net_G(input)
        # dis.append(net_D(output).cpu().numpy().item())
        reg = torch.zeros_like(latent)
        prob.append((reg_loss(latent.detach(), reg)).cpu().numpy().item())
        true.append(abs(label[0]))

fpr, tpr, thresholds = metrics.roc_curve(y_true = true, y_score = prob)

auc_score = metrics.auc(fpr, tpr)

# fpr2, tpr2, thresholds = metrics.roc_curve(y_true = true, y_score = dis)
# print(metrics.auc(fpr2, tpr2))
# print(dis)
plt.plot(fpr, tpr, 'o-')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive rate     '+str(auc_score))
plt.ylabel('True Positive rate')
plt.title('ROC Curve for '+ dataset)
# plt.draw()
# fig = plt.gcf()
plt.savefig('Result/'+model_name+'/'+dataset+'.png')
plt.show()

# print(tp, tn, fp, fn)
# precision =(tp + tn)/(tp+tn+fp+fn)
# recall = tn/(tn+fn)
# print('precision : ', 100*(tp + tn)/(tp+tn+fp+fn), '%')
# print('recall : ', 100*tn/(tn+fn), '%')
# F1 = 2*(precision *recall)/(precision+recall)
# print('F1 score : ', F1*100, ' %')

print('Testing End')
