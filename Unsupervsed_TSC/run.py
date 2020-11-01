import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import os
from data_load import train_load, test_load
from model import *
import statistics as stats
# from data_load2 import train_load, test_load
# from data_load_class import train_load, test_load, real_test_load
from importlib import import_module

# def poissonLoss(input):


def main(filename = 'ElectricDevices', batch = 64, epoch = 1000, model_name='test'):

    time, train = train_load(filename ,batch= batch)
    total_batches = len(train)

    device = torch.device('cuda')

    print('Data Loaded')

    # net = basic_cnn.Net()
    net_G = naive.Net(input = time).to(device)
    net_D = naive.Discriminator(input = time).to(device)
    # net = mnist.Net().to(device)

    adversarial_loss = nn.BCEWithLogitsLoss().to(device)
    reconstruction_loss = nn.MSELoss().to(device)
    reg_loss = nn.MSELoss().to(device)

    # criterion = F.nll_loss()
    # criterion = nn.Focal

    # net = torch.load('resnet.pth')

    # optimizer = optim.SGD(net.parameters(), lr = 1e-2, momentum=(0.5))
    lr = 2e-4
    lr_g = lr
    ## training

    print('Training Start')

    os.makedirs('pth/'+model_name, exist_ok=True)
    pth_name = 'pth/'+model_name+'/' + filename + '.pth'
    torch.save(net_G, pth_name)
    for epo in range(epoch):
        if epo % 100 == 0:
            lr_g = lr_g / 2
            print('***************************')
            print('learning rate : ', lr_g)
            print('***************************')
        # Optimizers
        optimizer_G = optim.Adam(net_G.parameters(), lr = lr_g)
        optimizer_D = optim.Adam(net_D.parameters(), lr = lr_g)

        for i, data in enumerate(train):

            input = data.to(device)
            output, latent = net_G(input)
            batches_done = epo * total_batches + i

            # Train Discriminator
            if batches_done % 3 == 0:
                optimizer_D.zero_grad()

                pred_real = net_D(input)
                pred_fake = net_D(output.detach())
                # valid = torch.ones_like(pred_real)
                # fake = torch.zeros_like(pred_fake)

                # real_loss = adversarial_loss(pred_real, valid)
                # fake_loss = adversarial_loss(pred_fake, fake)
                # real_loss = adversarial_loss(valid, pred_real)
                # fake_loss = adversarial_loss(fake,pred_fake)

                # d_loss = (real_loss + fake_loss) / 2
                d_loss = torch.mean(pred_fake) - torch.mean(pred_real)

                d_loss.backward()
                optimizer_D.step()

                for p in net_D.parameters():
                    p.data.clamp_(-0.01,0.01)

            # Train Generator

            optimizer_G.zero_grad()
            output, latent = net_G(input)
            pred_real = net_D(input).detach()
            pred_fake = net_D(output)
            # valid = torch.ones_like(pred_real)
            # fake = torch.zeros_like(pred_fake)
            # print(pred_fake-pred_real, valid.size())
            reg = torch.zeros_like(latent)
            # reg = torch.zeros_like(latent.mean())

            g_loss1 = -torch.mean(pred_fake)
            # g_loss1 = adversarial_loss(pred_fake, valid)
            g_loss2 = reconstruction_loss(output, input)
            # g_loss1 = adversarial_loss(valid, pred_fake)
            # add = torch.ones_like(latent)
            # latent = torch.add(latent, add)
            g_loss3 = reg_loss(reg, latent)*40
            # g_loss2 = 0
            g_loss = 0 * g_loss1 + 1 * g_loss2 +  0 * g_loss3

            g_loss.backward()
            optimizer_G.step()

            if batches_done % 200 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D Loss : %f] [G Loss : %f] [recon Loss : %f] [latent Loss : %f]"
                    % (epo, epoch, i, total_batches, d_loss.item(), g_loss.item(), g_loss2.item(), g_loss3.item())
                )
                # print(real_loss, fake_loss, pred_fake)
                # print(pred_fake)
                t , val =  train_load(filename ,batch= 1)
                # t , val, val_label =  test_load(filename)

                confidence = []
                confidence_ab = []

                for _, check in enumerate(val):
                    # print(check.to(device))
                    confidence.append(net_D(check.to(device)).detach().cpu().numpy().item())
                    val2, latent = net_G(check.to(device))
                    D_val2 = net_D(val2)
                    conf2 = abs(1 - D_val2.detach().cpu().numpy().item())
                    confidence_ab.append(conf2)
                # print(confidence)
                print('confidence : ', stats.mean(confidence))
                print('confidence_ab : ', stats.mean(confidence_ab))


            if batches_done % 200 == 0:
                time, test, test_label = test_load(filename)

                tp = 0
                tn = 0
                fp = 0
                fn = 0
                prob_0 = []
                prob_1 = []

                with torch.no_grad():
                    for i in range(len(test)):
                        input = test[i].to(device)
                        label = test_label[i]
                        output, latent = net_G(input)
                        reg = torch.zeros_like(latent)
                        if label[0] == 0:
                            prob_0.append((reg_loss(latent.detach(), reg)).cpu().numpy().item())
                        if label[0] == 1:
                            prob_1.append((reg_loss(latent.detach(), reg)).cpu().numpy().item())
                        # print(prob)
                    print('about norm, mean : ',stats.mean(prob_0), 'std : ',stats.stdev(prob_0))
                    print('about anomaly, mean : ',stats.mean(prob_1), 'std : ',stats.stdev(prob_1))
                D_pth_name = 'pth/'+model_name+'/' + filename + '_D' + '.pth'
                G_pth_name = 'pth/'+model_name+'/' + filename + '_G' + '.pth'
                torch.save(net_D, D_pth_name)
                torch.save(net_G, G_pth_name)

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

        # print('batch ', batch, ' epoch :  ', epo+1)

