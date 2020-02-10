from DataLoad import DataLoad
from DataLoad import load_file_list
from DataSet import CatDogDataSet
from DataSet import CatDogDataSet_files
from train_test import train
from train_test import test

from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

import torch.optim as optim

import numpy as np
import os

from Model_cut import load_object
from Model_cut import save_object

from ResNet import my_ResNet

def save():
    sampled = False
    run_name = 'trainable_worker50'
    path = 'model/ResNet_paras_{}.pkl'.format(run_name)
    net = my_ResNet(requeires_grad=False)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(path, map_location='cpu'))

    _, _, images_val_list, labels_val, _, _ = load_file_list(sampled=sampled)

    criterion = nn.CrossEntropyLoss()

    prediction = test(net, criterion, images_val_list, is_test=True, sampled=sampled, is_file_type=True,
                      model_type='resnet', my_path='./train')

    prediction_dict = {}
    for i in range(len(images_val_list)):
        print('{}, label = {}, prediction = {}'.format(images_val_list[i], 1 - labels_val[i], prediction[i]))
        prediction_dict[images_val_list[i]] = (1 - labels_val[i], prediction[i])

    if not os.path.exists('val_analysis'):
        os.makedirs('val_analysis')

    save_object(prediction_dict, 'val_analysis/val_prediction_result.pkl')

if __name__ == '__main__':
    # save()

    prediction_dict = load_object('val_analysis/val_prediction_result.pkl')
    print(prediction_dict)

    cnt = 0
    cnt_0 = 0

    for item in prediction_dict:
        if abs(prediction_dict[item][0] - prediction_dict[item][1]) > 0.5:
            print(item, prediction_dict[item])
            cnt += 1
            if prediction_dict[item][0] == 0:
                cnt_0 += 1

    print('total failed num: {}'.format(cnt))
    print('cnt_0 = {}'.format(cnt_0))