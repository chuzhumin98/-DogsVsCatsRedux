from DataLoad import DataLoad
from DataSet import CatDogDataSet
from train_test import train
from train_test import test

from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import numpy as np

class CNN_net(nn.Module):

    def __init__(self):
        super(CNN_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=1) #out: [batch_size, 6, 222, 222]
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=1) #out: [batch_size, 16, 72, 72]
        self.pool2 = nn.MaxPool2d(3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1,
                               padding=2)  # out: [batch_size, 32, 24, 24]
        self.pool3 = nn.MaxPool2d(3)

        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)  # reshape the input shape to [batch_size, 3, 224, 224]
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 32*8*8) # reshape to the flat layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    loader = DataLoad(sampled=False, model_type='cnn')
    images_train, labels_train, images_val, labels_val, image_test, test_filenames = loader.get_data()
    images_train = torch.from_numpy(images_train)
    labels_train = torch.from_numpy(labels_train)
    images_val = torch.from_numpy(images_val)
    labels_val = torch.from_numpy(labels_val)
    image_test = torch.from_numpy(image_test)

    print(images_train)
    print(labels_train)

    batch_size = 50
    class_num = 2
    num_workers = 30

    learning_rate = 0.01
    momentum = 0.9

    train_dataset = CatDogDataSet(images_train, labels_train)

    net = CNN_net()

    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    train(net, train_dataset, images_val, labels_val, batch_size, num_workers, criterion, optimizer,
          plot=True, plot_accuracy_name='image/accuracy_CNN_2.png', plot_loss_name='image/loss_CNN_2.png', plot_type='CNN')

    prediction = test(net, criterion, image_test, is_test=True)
    print(prediction)

    prediction_dict = {}
    for i in range(len(test_filenames)):
        id = int(test_filenames[i].split('.', 1)[0])
        prediction_dict[id] = prediction[i]

    file_out = open('result_CNN_2.csv', 'w')
    file_out.write('id,label\n')
    for key in sorted(prediction_dict.keys()):
        file_out.write('{},{}\n'.format(key, prediction_dict[key]))

    file_out.close()



