import torch
import csv
import numpy as np
import random

def train_load(filename='Wafer', batch=2):
    dataset = []
    f = open('data/' + filename + '/' + filename + '_TRAIN.tsv', 'r', encoding='utf-8', newline='')
    csvReader = csv.reader(f, delimiter=',')
    for row in csvReader:
        # print(row[2:])
        if row[0] == '':
            continue
        data=list(map(float, row[2:]))
        dataset.append([data])
    random.shuffle(dataset)
    f.close()
    time = len(dataset[0][0])
    train = np.array(dataset)
    if batch == 1:
        # train = torch.tensor(train).float()
        trainer = []
        for _, dat in enumerate(train):
            trainer.append([dat])
        train = trainer
        train = torch.tensor(train).float()
    else:
        num = 0
        trainer = []
        while num + batch < len(train):
            trainer.append(train[num:num + batch])
            num += batch
        train = trainer
        train = torch.tensor(train).float()

    return time, train

def test_load(filename = 'Wafer'):
    test = []
    test_label = []
    f = open( 'data/' + filename + '/' + filename + '_TEST.tsv', 'r', encoding='utf-8', newline='' )
    csvReader = csv.reader(f, delimiter = ',')
    for row in csvReader:
        if row[0] == '':
            continue
        data=list(map(float, row[1:]))
        test_label.append([int(data[0])])
        test.append([[data[1:]]])
    f.close()
    test = np.array(test)
    test = torch.tensor(test).float()
    time = len(test[0])
    return time, test, test_label

# print(train_load())
# print(test_load())
a,b = train_load('ECGFiveDays', batch=16)
print(a,b.size())
