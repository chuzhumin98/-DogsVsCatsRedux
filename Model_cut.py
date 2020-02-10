import torch.nn as nn
from torchvision import models

from DataLoad import DataLoad
import torch
from torch.autograd import Variable

import numpy as np
import pickle as pkl

import os

args = {
    'use_gpu': True
}

def load_object(path):
    with open(path,'rb') as f:
        obj = pkl.load(f)
    return obj

def save_object(obj,path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        # 取掉model的后两层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])

        self.transion_layer = nn.ConvTranspose2d(2048, 2048, kernel_size=14, stride=3)
        self.pool_layer = nn.MaxPool2d(32)
        self.Linear_layer = nn.Linear(2048, 8)

    def forward(self, x):
        x = self.resnet_layer(x)

        x = self.transion_layer(x)

        x = self.pool_layer(x)

        x = x.view(x.size(0), -1)

        x = self.Linear_layer(x)

        return x


class VGG_embedding(nn.Module):
    def __init__(self):
        super(VGG_embedding, self).__init__()
        self.net = models.vgg16(pretrained=True)
        self.net.classifier = nn.Sequential(*list(self.net.classifier.children())[:-3])

    def forward(self, x):
        return self.net(x)

class ResNet_embedding(nn.Module):
    def __init__(self):
        super(ResNet_embedding, self).__init__()
        # 取掉model的后两层
        net_init = models.resnet18(pretrained=True)
        self.net = nn.Sequential(*list(net_init.children())[:-1])

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 512)
        return x


def do_embedding(net, images, output_size, batch_size=50):
    print('start to do embedding')
    net.eval()
    n = images.shape[0]
    images_embedding = np.zeros([n, output_size])
    for i in range(n // batch_size):
        images_batch = images[i*batch_size:(i+1)*batch_size, :, :, :]
        images_batch = Variable(images_batch).type(torch.FloatTensor)

        if torch.cuda.is_available() and args['use_gpu']:
            images_batch = images_batch.cuda()

        out_batch = net(images_batch)

        images_embedding[i*batch_size:(i+1)*batch_size, :] = out_batch.cpu().data.numpy()

        print('{} of {}'.format(i, n // batch_size))

    return images_embedding

if __name__ == '__main__':
    '''
    model = models.resnet18(pretrained=True)
    print(model)
    net = Net(model)
    print(net)
    '''


    # net = VGG_embedding()
    net = ResNet_embedding()
    print(net)

    if torch.cuda.is_available() and args['use_gpu']:
        net = net.cuda()

    loader = DataLoad(sampled=False, model_type='resnet')
    images_train, labels_train, images_val, labels_val, images_test, test_filenames = loader.get_data(shuffled=False)
    images_train = torch.from_numpy(images_train)
    images_val = torch.from_numpy(images_val)
    images_test = torch.from_numpy(images_test)

    batch_size = 4

    output_size = 512 #512 vgg:4096, resnet:512

    images_train_embedded = do_embedding(net, images_train, output_size, batch_size=batch_size)
    images_val_embedded = do_embedding(net, images_val, output_size, batch_size=batch_size)
    images_test_embedded = do_embedding(net, images_test, output_size, batch_size=batch_size)

    embedding_resnet = {'train': (images_train_embedded, labels_train), 'val': (images_val_embedded, labels_val),
                        'test': (images_test_embedded, test_filenames)}

    if not os.path.exists('cache'):
        os.makedirs('cache')

    save_object(embedding_resnet, 'cache/embedding_resnet_nonshuffle.pkl')




