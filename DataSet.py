import numpy as  np
import torch

from PIL import Image
import os

class CatDogDataSet:
    def __init__(self, data, labels):
        self.x = data
        self.y = labels


    def __getitem__(self, item):
        return self.x[item,:,:,:], self.y[item]

    def __len__(self):
        return len(self.y)


class CatDogDataSet_files:
    def __init__(self, file_names, labels, sampled=False, width=224, height=224, model_type='cnn'):
        self.files = file_names
        self.labels = labels
        if sampled:
            self.path_train = 'train_sample'
        else:
            self.path_train = 'train'
        self.width = width
        self.height = height
        self.model_type = model_type

    def __getitem__(self, item):
        image_RGB = np.zeros([3, self.width, self.height])
        fn = self.files[item]
        im = Image.open(os.path.join(self.path_train, fn))

        im_resize = im.resize((self.width, self.height))

        image = np.asarray(im_resize)

        if self.model_type == 'resnet':
            means = [0.485, 0.456, 0.406]
            stds = [0.229, 0.224, 0.225]
        else:
            means = [0.5, 0.5, 0.5]
            stds = [0.5, 0.5, 0.5]

        for j in range(3):
            mean = np.mean(image[:, :, j])
            std = np.std(image[:, :, j])
            image_RGB[j, :, :] = stds[j] * (image[:, :, j] - mean) / std + means[j]

        return image_RGB, self.labels[item]

    def __len__(self):
        return len(self.labels)


def get_image_item(path, file_name, width=224, height=224, model_type='cnn'):
    image_RGB = np.zeros([3, width, height])
    im = Image.open(os.path.join(path, file_name))

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
        image_RGB[j, :, :] = stds[j] * (image[:, :, j] - mean) / std + means[j]

    return image_RGB