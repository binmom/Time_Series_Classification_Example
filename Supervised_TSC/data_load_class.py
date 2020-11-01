import torch
import csv
import numpy as np
import torch.utils.data as utils

def train_load(filename = 'ElectricDevices', batch = 2):
    train = []
    train_label = []
    f = open( 'data/' + filename + '/' + filename + '_TRAIN.tsv', 'r', encoding='utf-8', newline='' )
    csvReader = csv.reader(f, delimiter = '\t')
    for row in csvReader:
        # print(len(row))
        data = row
        data = list(map(float, data))
        train_label.append(int(data[0]-1))
        train.append([data[1:]])
    f.close()
    cls = max(train_label)+1
    if min(train_label)==-1:
        cls = max(train_label)+2
        train_label = [x+1 for x in train_label]
    elif min(train_label)==-2:
        cls = max(train_label)+3
        train_label = [x+2 for x in train_label]
    time = len(train[0][0])


    
    train = np.array(train)
    train_label = np.array(train_label)
    if batch == 1:
        train = torch.tensor(train).float()
        train_label = torch.tensor(train_label)
    else:
        num = 0
        trainer = []
        trainer_label = []
        while num+batch < len(train):
            trainer.append(train[num:num+batch])
            trainer_label.append(train_label[num:num+batch])
            num += batch
        train = trainer
        train_label = trainer_label
        train = torch.tensor(train).float()
        train_label = torch.tensor(train_label)
        # print(cls)
    return cls, time, train, train_label

def test_load(filename = 'ElectricDevices'):
    test = []
    test_label = []
    f = open( 'data/' + filename + '/' + filename + '_TEST.tsv', 'r', encoding='utf-8', newline='' )
    csvReader = csv.reader(f, delimiter = '\t')
    for row in csvReader:
        if row[1] == 'class':
            continue
        data = row
        data = list(map(float, data))
        test_label.append([int(data[0]-1)])
        test.append([[data[1:]]])
    f.close()
    test = np.array(test)
    test = torch.tensor(test).float()
    return test, test_label

def real_test_load(filename = 'ElectricDevices'):
    test = []
    test_label = []
    f = open( 'data/' + filename + '/' + filename + '_TEST.tsv', 'r', encoding='utf-8', newline='' )
    csvReader = csv.reader(f, delimiter = '\t')
    for row in csvReader:
        if row[1] == 'class':
            continue
        data = row
        data = list(map(float, data))
        test_label.append([int(data[0]-1)])
        test.append([[data[1:]]])
    f.close()
    test = np.array(test)
    test = torch.tensor(test).float()
    return test, test_label

# train_load()
