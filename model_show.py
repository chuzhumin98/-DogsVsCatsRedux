from torchvision import models

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pickle as pkl
from torch.autograd import Variable
import torch

import os
import sys

import math

from ResNet import my_ResNet

def data_preprocess(path, width=224, height=224, model_type='resnet'):
    images_RGB = np.random.randn(1, 3, width, height) # np.zeros([50, 3, width, height])

    im = Image.open(path)

    im_resize = im.resize((width, height))

    image = np.asarray(im_resize)

    if model_type == 'resnet':
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
    else:
        means = [0.5, 0.5, 0.5]
        stds = [0.5, 0.5, 0.5]


    for j in range(3):
        mean = np.mean(image[:, :, j])
        std = np.std(image[:, :, j])
        images_RGB[0, j, :, :] = stds[j] * (image[:, :, j] - mean) / std + means[j]

    # for i in range(1,50):
     #   images_RGB[i, :, :, :] = images_RGB[0, :, :, :]


    images_RGB = torch.from_numpy(images_RGB)

    # print(images_RGB[0, :, :, :])

    return images_RGB, im


if __name__ == '__main__':
    path = 'model/ResNet_paras_3.pkl'

    if torch.cuda.is_available():
        net = my_ResNet(requeires_grad=False)
        net.load_state_dict(torch.load(path))
    else:
        net = my_ResNet(requeires_grad=False)
        net.load_state_dict(torch.load(path, map_location='cpu'))

    print(net)

    net.eval()

    test_files = ['./me/test_1.jpeg', './me/test_2.png', './me/test_3.jpeg', './me/test_4.png']

    idx = 2

    # image, im = data_preprocess('./test/5000.jpg')
    image, im = data_preprocess(test_files[idx])
    image = image.type(torch.FloatTensor)
    # image = Variable(image).type(torch.FloatTensor)

    if torch.cuda.is_available():
        image = image.cuda()

    prediction = net(image).data.numpy()

    print(prediction)

    type = 'dog'
    if prediction[0, 0] > prediction[0, 1]:
        print('This is a dog')
    else:
        print('This is a cat')
        type = 'cat'

    prob = math.exp(np.max(prediction[0, :])) / np.sum(np.exp(prediction[0, :]))

    print('the confidence prob is {}'.format(prob))

    fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 40)
    d = ImageDraw.Draw(im)
    if idx != 2:
        d.text((10, 10), "This is a {} with prob {}".format(type, prob), font=fnt, fill=(255, 255, 0))
    else:
        d.text((0, 10), "This is a {} with prob {}".format(type, prob),
               font=ImageFont.truetype('/Library/Fonts/Arial.ttf', 20), fill=(255, 0, 255))

    im.show()
