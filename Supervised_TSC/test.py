import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from data_load_class import train_load, test_load, real_test_load
from model import *



# print('Data Loaded')

# model = torch.load('basic_focal_loss.pth')
# model = torch.load('basic_10.pth')



def test():
    O = 0
    X = 0
    test, test_label = real_test_load()

    device = torch.device('cuda')

    model = torch.load('pth/r.pth')

    with torch.no_grad():
        for i in range(len(test)):
            input = test[i].to(device)
            label = test_label[i]
            output = model(input).cpu()
            # print(torch.max(output,0))
            _, index = torch.max(output, 0)
            # print(_, index)
            index = index.numpy()
            # print(index,label)
            label = label[0]
            if label == index:
                O += 1
            else:
                X += 1
            # else:
        #     print(label, index)

    acc = O / (O + X)
    print(acc)
    return acc

# tp = 0
# tn = 0
# fp = 0
# fn = 0
#
# with torch.no_grad():
#     for i in range(len(test)):
#     # for i in range(1):
#         input = test[i]
#         label = test_label[i]
#         output = model(input).numpy()[0][0]
#         # print(output)
#         if round(output) ==0 and label==0:
#             tp +=1
#         elif round(output) ==1 and label ==0:
#             fp +=1
#         elif round(output) == 0 and label == 1:
#             fn +=1
#         elif round(output) == 1 and label == 1:
#             tn +=1
#         else:
#             print(output, label)
# print(tp, tn, fp, fn)
# precision =(tp + tn)/(tp+tn+fp+fn)
# recall = tn/(tn+fn)
# print('precision : ', 100*(tp + tn)/(tp+tn+fp+fn), '%')
# print('recall : ', 100*tn/(tn+fn), '%')
# F1 = 2*(precision *recall)/(precision+recall)
# print('F1 score : ', F1*100, ' %')
