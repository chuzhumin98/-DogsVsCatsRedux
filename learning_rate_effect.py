from DataLoad import load_object

import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    path_prefix = 'tables/ResNet_table_lr'
    learning_rates = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
    learning_rates_str = [str(learning_rate) for learning_rate in learning_rates]

    models_num = len(learning_rates)
    size = 40

    train_accus = np.zeros([models_num, size])
    val_accus = np.zeros([models_num, size])
    train_losses = np.zeros([models_num, size])

    for i in range(models_num):
        data = load_object('{}{}.pkl'.format(path_prefix, learning_rates_str[i]))
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
    plt.title('train accuracy vs epoch in different learning rate')
    plt.legend(learning_rates_str)
    plt.savefig('image/train_accu_lrs.png', dpi=150)

    plt.figure(1)
    for i in range(models_num):
        plt.plot(xs, val_accus[i, :])
    plt.xlabel('epoch')
    plt.ylabel('validate accuracy')
    plt.title('validate accuracy vs epoch in different learning rate')
    plt.legend(learning_rates_str)
    plt.savefig('image/val_accu_lrs.png', dpi=150)

    plt.figure(2)
    for i in range(models_num):
        plt.plot(xs, train_losses[i, :])
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.title('train loss vs epoch in different learning rate')
    plt.legend(learning_rates_str)
    plt.savefig('image/train_loss_lrs.png', dpi=150)

    plt.figure(3)
    xs = np.array(range(models_num))
    best_val_accu = np.max(val_accus, axis=1)
    plt.plot(xs, best_val_accu, 'b', marker='.', markersize=10, lw=1.5)
    for i in range(len(xs)):
        if i == 0:
            plt.text(xs[i] + 0.04, best_val_accu[i] - 0.0001, str(best_val_accu[i]))
        elif i == 2:
            plt.text(xs[i] + 0.04, best_val_accu[i] - 0.00015, str(best_val_accu[i]))
        else:
            plt.text(xs[i] + 0.04, best_val_accu[i] + 0.00004, str(best_val_accu[i]))
    plt.xticks(xs, learning_rates_str)
    plt.title('best validate accuracy vs learning rate')
    # plt.grid()
    plt.savefig('image/best_val_accu_lr.png')

    plt.show()


