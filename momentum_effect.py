from DataLoad import load_object

import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    path_prefix = 'tables/ResNet_table_momentum'
    momentums = [0, 0.1, 0.5, 0.9]
    momentums_str = [str(momentum) for momentum in momentums]

    models_num = len(momentums)
    size = 40

    train_accus = np.zeros([models_num, size])
    val_accus = np.zeros([models_num, size])
    train_losses = np.zeros([models_num, size])

    for i in range(models_num):
        data = load_object('{}{}.pkl'.format(path_prefix, momentums_str[i]))
        train_accus[i, :] = data['train_accu']
        val_accus[i, :] = data['val_accu']
        train_losses[i, :] = data['loss']


    xs = 0.5 * np.array(range(1, size+1))

    if not os.path.exists('image'):
        os.makedirs('image')

    plt.figure(0)
    for i in range(models_num):
        plt.plot(xs, train_accus[i, :])
    plt.xlabel('epoch')
    plt.ylabel('train accuracy')
    plt.title('train accuracy vs epoch in different momentum')
    plt.legend(momentums_str)
    plt.savefig('image/train_accu_momentums.png', dpi=150)

    plt.figure(1)
    for i in range(models_num):
        plt.plot(xs, val_accus[i, :])
    plt.xlabel('epoch')
    plt.ylabel('validate accuracy')
    plt.title('validate accuracy vs epoch in different momentum')
    plt.legend(momentums_str)
    plt.savefig('image/val_accu_momentums.png', dpi=150)

    plt.figure(2)
    for i in range(models_num):
        plt.plot(xs, train_losses[i, :])
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.title('train loss vs epoch in different momentum')
    plt.legend(momentums_str)
    plt.savefig('image/train_loss_momentums.png', dpi=150)

    plt.figure(3)
    xs = np.array(range(models_num))
    best_val_accu = np.max(val_accus, axis=1)
    plt.plot(xs, best_val_accu, 'b', marker='.', markersize=10, lw=1.5)
    for i in range(len(xs)):
        plt.text(xs[i] + 0.04, best_val_accu[i] - 0.0001, str(best_val_accu[i]))
    plt.xticks(xs, momentums_str)
    plt.title('best validate accuracy vs momentum')
    # plt.grid()
    plt.savefig('image/best_val_accu_momentums.png')

    plt.show()


