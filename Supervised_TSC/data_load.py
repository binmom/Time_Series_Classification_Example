import torch
import csv
import numpy as np
import random


def train_load(filename='ElectricDevices', batch=2):
    train = []
    train_label = []
    dataset = []
    f = open('data/' + filename + '/' + filename + '_TRAIN.tsv', 'r', encoding='utf-8', newline='')
    csvReader = csv.reader(f, delimiter='\t')
    for row in csvReader:
        dataset.append(list(map(float, row)))
    random.shuffle(dataset)
    num = len(dataset)
    train_data = dataset[:round(num*0.8)]
    val_data = dataset[round(num*0.8):]
    for data in train_data:
        train_label.append(int(data[0] - 1))
        train.append([data[1:]])
    f.close()
    cls = max(train_label) + 1
    if min(train_label) == -1:
        cls = max(train_label) + 2
        train_label = [x + 1 for x in train_label]
    elif min(train_label) == -2:
        cls = max(train_label) + 2
        for i in range(len(train_label)):
            if train_label[i] == -2:
                train_label[i] += 3
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
        while num + batch < len(train):
            trainer.append(train[num:num + batch])
            trainer_label.append(train_label[num:num + batch])
            num += batch
        train = trainer
        train_label = trainer_label
        train = torch.tensor(train).float()
        train_label = torch.tensor(train_label)
        # print(cls)

    val = []
    val_label=[]

    for data in val_data:
        val_label.append([int(data[0] - 1)])
        val.append([[data[1:]]])
    val = np.array(val)
    val = torch.tensor(val).float()

    print(len(train_label)*batch,len(val_label))

    return cls, time, train, train_label, val, val_label

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
    if min(test_label) == -1:
        test_label = [x + 1 for x in test_label]
    elif min(test_label) == -2:
        for i in range(len(test_label)):
            if test_label[i] == -2:
                test_label[i] += 3
    f.close()
    test = np.array(test)
    test = torch.tensor(test).float()
    print(len(test))
    return test, test_label