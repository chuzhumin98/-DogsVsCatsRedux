import os

import numpy as np
from Model_cut import save_object
from Model_cut import load_object


def is_cat(label):
    if label == 'cat':
        return 1
    else:
        return 0

def generate_list(path, is_train=True, is_sample=True):
    if not os.path.exists('file_list'):
        os.makedirs('file_list')

    for dirpath, dirnames, filenames in os.walk(path):
        if is_train:
            index_array = np.array(range(len(filenames)))
            np.random.shuffle(index_array)

            labels_list = [is_cat(fn[:3]) for fn in filenames]

            filenames_shuffle = [filenames[idx] for idx in index_array]
            labels_shuffle = [labels_list[idx] for idx in index_array]

            data = {'file': filenames_shuffle, 'label': labels_shuffle}
            print(labels_shuffle)

            if is_sample:
                save_object(data, 'file_list/train_sample_list.pkl')
            else:
                save_object(data, 'file_list/train_list.pkl')

        else:
            index_array = np.array(range(len(filenames)))
            np.random.shuffle(index_array)

            filenames_shuffle = [filenames[idx] for idx in index_array]

            data = {'file': filenames_shuffle}

            if is_sample:
                save_object(data, 'file_list/test_sample_list.pkl')
            else:
                save_object(data, 'file_list/test_list.pkl')






if __name__ == '__main__':
    path, is_train, is_sample = './train_sample', True, True
    path, is_train, is_sample = './train', True, False

    path, is_train, is_sample = './test_sample', False, True
    path, is_train, is_sample = './test', False, False

    generate_list(path, is_train=is_train, is_sample=is_sample)