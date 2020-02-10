from DataLoad import save_object
from DataLoad import load_object

from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

import torch.optim as optim

import os

from train_test import train
from train_test import test

import numpy as np

def get_data(path_resnet_embed, path_vgg_embed):
    resnet_embedding = load_object(path_resnet_embed)
    vgg_embedding = load_object(path_vgg_embed)
    images_train = np.hstack((resnet_embedding['train'][0], vgg_embedding['train'][0]))
    labels_train = resnet_embedding['train'][1]
    images_val = np.hstack((resnet_embedding['val'][0], vgg_embedding['val'][0]))
    labels_val = resnet_embedding['val'][1]
    images_test = np.hstack((resnet_embedding['test'][0], vgg_embedding['test'][0]))
    test_filenames = resnet_embedding['test'][1]
    print(images_train.shape)

    return images_train, labels_train, images_val, labels_val, images_test, test_filenames




class EmbeddingDataSet:
    def __init__(self, total_embedding, labels):
        self.x = total_embedding
        self.y = labels


    def __getitem__(self, item):
        return self.x[item, :], self.y[item]

    def __len__(self):
        return len(self.y)


class Mixture_Model(nn.Module):
    def __init__(self, hidden_num=50, embedding_number_resnet=512, embedding_number_vgg=4096):
        super(Mixture_Model, self).__init__()
        self.embedding_number_resnet = embedding_number_resnet
        self.embedding_number_vgg = embedding_number_vgg
        self.fc_resnet = nn.Linear(embedding_number_resnet, hidden_num)
        self.fc_vgg = nn.Linear(embedding_number_vgg, hidden_num)
        self.fc1 = nn.Linear(hidden_num*2, 2)


    def forward(self, input):
        x_resnet = F.relu(self.fc_resnet(input[:, :self.embedding_number_resnet]))
        x_vgg = F.relu(self.fc_vgg(input[:, self.embedding_number_resnet:]))
        x = torch.cat((x_resnet, x_vgg), 1)
        x = self.fc1(x)
        return x

def main(sampled = False, run_name='', hidden_num=50):
    batch_size = 50
    class_num = 2
    num_workers = 20

    learning_rate = 0.001 # default 0.001
    momentum = 0.9  # default 0.9

    net = Mixture_Model(hidden_num=hidden_num)

    images_train, labels_train, images_val, labels_val, images_test, test_filenames = \
        get_data('cache/embedding_resnet_nonshuffle.pkl', 'cache/embedding_vgg_nonshuffle.pkl')

    images_train = torch.from_numpy(images_train)
    images_val = torch.from_numpy(images_val)
    images_test = torch.from_numpy(images_test)
    labels_train = torch.from_numpy(labels_train)
    labels_val = torch.from_numpy(labels_val)

    train_dataset_embedding = EmbeddingDataSet(images_train, labels_train)

    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    if not os.path.exists('image'):
        os.makedirs('image')

    # val_accu = test(net, criterion, images_val, labels=labels_val, is_test=False)
    # print('val_accu = {}'.format(val_accu))

    accuracy_list, validate_accuracy_list, loss_list = train(net, train_dataset_embedding, images_val, labels_val, batch_size,
                                                             num_workers, criterion, optimizer,
                                                             plot=True,
                                                             plot_accuracy_name='image/accuracy_Mixture_{}.png'.format(
                                                                 run_name),
                                                             plot_loss_name='image/loss_Mixture_{}.png'.format(
                                                                 run_name), plot_type='Mixture Model',
                                                             save=True,
                                                             save_name='./model/Mixture_paras_{}.pkl'.format(
                                                                 run_name),
                                                             dim=2)


    if not os.path.exists('tables'):
        os.makedirs('tables')

    file_name = 'Mixture_table_{}.pkl'.format(run_name)
    table_dict = {'num_workers': num_workers, 'train_accu': accuracy_list, 'val_accu': validate_accuracy_list,
                  'loss': loss_list}
    save_object(table_dict, os.path.join('tables', file_name))



def predict_with_best_net(run_name, sampled=False, hidden_num=50):
    path = 'model/Mixture_paras_{}.pkl'.format(run_name)
    net = Mixture_Model(hidden_num=hidden_num)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(path, map_location='cpu'))

    _, _, _, _, images_test, test_filenames = get_data('cache/embedding_resnet_nonshuffle.pkl', 'cache/embedding_vgg_nonshuffle.pkl')

    images_test = torch.from_numpy(images_test)

    criterion = nn.CrossEntropyLoss()

    prediction = test(net, criterion, images_test, is_test=True, sampled=sampled, is_file_type=False, dim=2)

    prediction_dict = {}
    for i in range(len(test_filenames)):
        id = int(test_filenames[i].split('.', 1)[0])
        prediction_dict[id] = prediction[i]

    if not os.path.exists('result'):
        os.makedirs('result')

    file_out = open('./result/result_Mixture_{}.csv'.format(run_name), 'w')
    file_out.write('id,label\n')
    for key in sorted(prediction_dict.keys()):
        file_out.write('{},{}\n'.format(key, prediction_dict[key]))

    file_out.close()



if __name__ == '__main__':
    sampled = False
    run_name = 'hidden50'
    hidden_num = 50

    main(sampled=sampled, run_name=run_name, hidden_num=hidden_num)
    predict_with_best_net(run_name=run_name, sampled=sampled, hidden_num=hidden_num)

