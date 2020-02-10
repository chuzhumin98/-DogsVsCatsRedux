import os
import numpy as np

from PIL import Image
import pickle as pkl

class DataLoad:
    def __init__(self, eta=0.8, path_train='./train', path_test='./test', width=224, height=224, used_npz=False,
                 path_npz='./data/DATA.npz', sampled=True, model_type='cnn'):
        if sampled:
            path_train = './train_sample'
            path_test = './test_sample'
        self.path_train = path_train
        self.path_test = path_test
        self.width = width
        self.height = height
        self.used_npz = used_npz
        self.path_npz = path_npz
        self.eta = eta # the ratio of train data
        self.model_type = model_type

    def is_cat(self, label):
        if label == 'cat':
            return 1
        else:
            return 0

    def load_train_data(self, flat=False):
        print('start to load train data')
        for dirpath, dirnames, filenames in os.walk(self.path_train):
            train_labels = np.array([self.is_cat(fn[:3]) for fn in filenames])
            if not flat:
                images_RGB = np.zeros([len(filenames), 3, self.width, self.height])
                for i in range(len(filenames)):
                    fn = filenames[i]
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
                        images_RGB[i, j, :, :] = stds[j] * (image[:, :, j] - mean) / std + means[j]

                    # print(images_RGB[i, :, :, :])

                print('end to load train data')
                return images_RGB, train_labels
            else:
                import torch.nn as nn
                import torch
                pool = nn.MaxPool2d(4)

                images_L = np.zeros([len(filenames), self.width*self.height//16])
                for i in range(len(filenames)):
                    fn = filenames[i]
                    im = Image.open(os.path.join(self.path_train, fn))
                    im = im.convert('L')
                    im_resize = im.resize((self.width, self.height))

                    image = np.asarray(im_resize).astype(np.float32)
                    image = image.reshape([1, self.width, self.height])

                    image = pool(torch.from_numpy(image))
                    images_L[i, :] = image.numpy().reshape([-1])

                print('end to load train data')
                return images_L, train_labels


    def load_test_data(self, flat=False):
        print('start to load test data')
        for dirpath, dirnames, filenames in os.walk(self.path_test):
            if not flat:
                images_RGB = np.zeros([len(filenames), 3, self.width, self.height])
                for i in range(len(filenames)):
                    fn = filenames[i]
                    im = Image.open(os.path.join(self.path_test, fn))

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
                        images_RGB[i, j, :, :] = stds[j] * (image[:, :, j] - mean) / std + means[j]

                    # print(images_RGB[i, :, :, :])

                print('end to load test data')
                return images_RGB, filenames
            else:
                import torch.nn as nn
                import torch
                pool = nn.MaxPool2d(4)

                images_L = np.zeros([len(filenames), self.width * self.height // 16])
                for i in range(len(filenames)):
                    fn = filenames[i]
                    im = Image.open(os.path.join(self.path_test, fn))
                    im = im.convert('L')
                    im_resize = im.resize((self.width, self.height))

                    image = np.asarray(im_resize).astype(np.float32)
                    image = image.reshape([1, self.width, self.height])

                    image = pool(torch.from_numpy(image))
                    images_L[i, :] = image.numpy().reshape([-1])

                print('end to load test data')
                return images_L, filenames

    def get_data(self, flat=False, shuffled=True):
        if not self.used_npz:
            images, labels = self.load_train_data(flat)
            images_test, test_filenames = self.load_test_data(flat)
            # np.savez('./data/DATA.npz', images=images, labels=labels, images_test=images_test)
        else:
            data = np.load(self.path_npz)
            images = np.array(data['images'])
            labels = np.array(data['labels'])
            images_test = np.array(data['image_test'])
            for dirpath, dirnames, filenames in os.walk(self.path_test):
                test_filenames = filenames



        print(images.shape, images_test.shape)

        index_array = np.array(range(len(labels)))
        if shuffled:
            np.random.shuffle(index_array)
        split_num = round(self.eta * len(labels))
        return images[index_array[:split_num]], labels[index_array[:split_num]], images[index_array[split_num:]], labels[index_array[split_num:]], images_test, test_filenames

def load_object(path):
    with open(path,'rb') as f:
        obj = pkl.load(f)
    return obj

def save_object(obj,path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)

def load_file_list(sampled=False, eta=0.8):
    if sampled:
        path_train_list = 'file_list/train_sample_list.pkl'
        path_test_list = 'file_list/test_sample_list.pkl'
    else:
        path_train_list = 'file_list/train_list.pkl'
        path_test_list = 'file_list/test_list.pkl'
    data = load_object(path_train_list)
    file_list, labels = data['file'], data['label']
    split_num = round(len(labels) * eta)
    images_train_list = file_list[:split_num]
    labels_train = np.array(labels[:split_num])
    images_val_list = file_list[split_num:]
    labels_val = np.array(labels[split_num:])

    data = load_object(path_test_list)
    image_test_list = data['file']
    return images_train_list, labels_train, images_val_list, labels_val, image_test_list, image_test_list


if __name__ == '__main__':
    table = load_object('tables/ResNet_table_momentum1.5.pkl')
    print(table)
    print('the best val accu: {}'.format(max(table['val_accu'])))
    print('the best train accu: {}'.format(max(table['train_accu'])))


    images_train_list, labels_train, images_val_list, labels_val, image_test_list, test_filenames = load_file_list(sampled=True)
    print(images_val_list)
    print(len(labels_val))
