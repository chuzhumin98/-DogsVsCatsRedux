from PIL import Image
import numpy as np

from DataLoad import DataLoad

from DataLoad import load_object
from DataLoad import save_object

import pickle as pkl

import matplotlib.pyplot as plt


if __name__ == '__main__':
    # path = 'train/cat.0.jpg'
    # im = Image.open(path)
    # print(im.size)
    # # im.show()
    # im_resize = im.resize((224, 224))
    # print(im_resize.size)
    # # im_resize.show()
    #
    # image = np.asarray(im_resize)
    # print(image)
    # print(image.shape)
    #
    # image_RGB = np.array([image[:,:,0], image[:,:,1], image[:,:,2]])
    # print(image_RGB)
    # print(image_RGB.shape)
    #
    # loader = DataLoad()
    # loader.get_data()

    # resnet_embedding = load_object('cache/embedding_resnet_nonshuffle.pkl')
    # train_labels_resnet = np.array(resnet_embedding['train'][1])
    # print(train_labels_resnet)
    #
    # vgg_embedding = load_object('cache/embedding_vgg_nonshuffle.pkl')
    # train_labels_vgg = np.array(vgg_embedding['train'][1])
    # print(train_labels_vgg)
    # print(np.sum(np.abs(train_labels_resnet - train_labels_vgg)))
    #
    #
    # print(resnet_embedding['test'][1])
    # print(vgg_embedding['test'][1])

    random_list = np.random.randn(100000)
    print(random_list)

    plt.hist(random_list, bins=20, color='r')
    plt.show()