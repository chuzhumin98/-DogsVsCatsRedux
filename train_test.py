from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import os

from DataSet import get_image_item

val_path_dict = {True: './train_sample', False: './train'}
test_path_dict = {True: './test_sample', False: './test'}
train_size_dict = {True: 800, False: 20000}

def train(net, train_dataset, images_val, labels_val, batch_size, num_workers, criterion, optimizer,
          plot=False, plot_accuracy_name='accuracy.png', plot_loss_name='loss.png', plot_type='Neural Network',
          save=False, save_name='./model/model.pkl', sampled=False, is_file_type=False, width=224, height=224,
          model_type='cnn', batch_val=50, dim=4):
    disp_perepoch = 2
    num_perdisp = train_size_dict[sampled] // batch_size // disp_perepoch

    net.train()
    best_accu = -1.
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    accuracy_list = np.zeros([num_workers*disp_perepoch]) # the list to store accuracy of each epoch
    validate_accuracy_list = np.zeros([num_workers*disp_perepoch])  # the list to store accuracy of each epoch
    loss_list = np.zeros([num_workers*disp_perepoch])

    if torch.cuda.is_available():
        net = net.cuda()

    cnt = 0

    for epoch in range(num_workers):
        for j in range(disp_perepoch):
            accuracys = []
            losses = []
            for i, data in enumerate(train_loader, j*num_perdisp):
                if i == (j+1) * num_perdisp:
                    break

                inputs, labels = data

                inputs = Variable(inputs).type(torch.FloatTensor)

                labels = Variable(labels).type(torch.LongTensor)

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # note that CrossEntropyLoss has done optimization, so the following 2 rows are useless
                # labels = labels.view(batch_size, 1)
                # labels = torch.zeros(batch_size, class_num).scatter_(1, labels, 1)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                accuracy = int((predicted == labels).sum()) / batch_size
                accuracys.append(accuracy)
                losses.append(loss.item())

                print('epoch {} index {} loss = {}, accuracy = {}'.format(epoch, i, loss.item(), accuracy))

            accuracy_list[cnt] = np.mean(accuracys)
            loss_list[cnt] = np.mean(losses)
            validate_accuracy_list[cnt] = test(net, criterion, images_val, labels_val, False,
                                                 sampled=sampled, is_file_type=is_file_type,
                                               width=width, height=height, model_type=model_type,
                                               batch_val=batch_val, dim=dim)

            print('epoch {} average train accuracy is {}, loss is {}'.format(epoch, accuracy_list[cnt], loss_list[cnt]))
            print('epoch {} average validate accuracy is {}'.format(epoch, validate_accuracy_list[cnt]))

            if best_accu < validate_accuracy_list[cnt]:
                best_accu = validate_accuracy_list[cnt]
                if save:
                    if not os.path.exists('./model'):
                        os.makedirs('./model')
                    torch.save(net.state_dict(), save_name)

            cnt += 1

    if plot:
        xs = np.array(range(len(accuracy_list)))
        plt.switch_backend('agg')
        plt.figure(0)
        plt.plot(xs, accuracy_list, 'b')
        plt.plot(xs, validate_accuracy_list, 'r')
        plt.legend(['Train', 'Validate'])
        plt.title('Accuracy vs Epoch in {}'.format(plot_type))
        plt.box(on=True)
        plt.savefig(plot_accuracy_name, dpi=150)

        plt.figure(1)
        plt.plot(xs, loss_list, 'b')
        plt.title('Loss vs Epoch in {}'.format(plot_type))
        plt.box(on=True)
        plt.savefig(plot_loss_name, dpi=150)

    return accuracy_list, validate_accuracy_list, loss_list

def test(net, criterion, inputs, labels=None, is_test=False, sampled=False, is_file_type=False,
         width=224, height=224, model_type='cnn', batch_val=50, dim=4, my_path=None):
    net.eval()
    if not is_test:
        if not is_file_type:
            inputs = Variable(inputs).type(torch.FloatTensor)
            labels = Variable(labels).type(torch.LongTensor)

        correct_num = 0
        for batch in range(len(labels) // batch_val):
            if not is_file_type:
                if dim == 4:
                    batch_inputs = inputs[batch*batch_val:(batch+1)*batch_val, :, :, :]
                elif dim == 2:
                    batch_inputs = inputs[batch * batch_val:(batch + 1) * batch_val, :]
            else:
                if my_path is not None:
                    path = my_path
                else:
                    path = val_path_dict[sampled]
                batch_inputs = np.zeros([batch_val, 3, width, height])
                for i in range(batch_val):
                    if dim == 4:
                        batch_inputs[i, :, :, :] = get_image_item(path, inputs[batch*batch_val+i], width, height, model_type=model_type)
                    elif dim == 2:
                        batch_inputs[i, :] = get_image_item(path, inputs[batch * batch_val + i], width, height,
                                                                  model_type=model_type)
                batch_inputs = torch.from_numpy(batch_inputs)
                batch_inputs = Variable(batch_inputs).type(torch.FloatTensor)

            batch_labels = labels[batch*batch_val:(batch+1)*batch_val]

            if torch.cuda.is_available():
                batch_inputs = batch_inputs.cuda()
                batch_labels = batch_labels.cuda()

            outputs = net(batch_inputs)

            # print(outputs.data)

            _, predicted = torch.max(outputs.data, 1)
            correct_num += int((predicted == batch_labels).sum())

        return correct_num / len(labels)
    else:
        if not is_file_type:
            inputs = Variable(inputs).type(torch.FloatTensor)

        outputs_total = np.zeros(len(inputs))
        for batch in range(len(inputs) // batch_val):
            if not is_file_type:
                if dim == 4:
                    batch_inputs = inputs[batch*batch_val:(batch+1)*batch_val, :, :, :]
                elif dim == 2:
                    batch_inputs = inputs[batch * batch_val:(batch + 1) * batch_val, :]
            else:
                if my_path is not None:
                    path = my_path
                else:
                    path = test_path_dict[sampled]
                batch_inputs = np.zeros([batch_val, 3, width, height])
                for i in range(batch_val):
                    if dim == 4:
                        batch_inputs[i, :, :, :] = get_image_item(path, inputs[batch*batch_val+i], width, height, model_type=model_type)
                    elif dim == 2:
                        batch_inputs[i, :] = get_image_item(path, inputs[batch * batch_val + i], width, height,
                                                                  model_type=model_type)
                batch_inputs = torch.from_numpy(batch_inputs)
                batch_inputs = Variable(batch_inputs).type(torch.FloatTensor)

            if torch.cuda.is_available():
                batch_inputs = batch_inputs.cuda()

            outputs = net(batch_inputs)
            outputs = outputs.cpu()
            outputs = outputs.data.numpy()
            # print(outputs)
            outputs = outputs - np.tile(np.min(outputs, axis=1).reshape(batch_val, 1), [1, 2])
            outputs = np.minimum(outputs, 20)
            outputs_total[batch*batch_val:(batch+1)*batch_val] = np.exp(outputs[:, 0]) / (np.exp(outputs[:, 0]) + np.exp(outputs[:, 1]))

        return outputs_total


def multirun(Network, train_dataset, validate_dataset, test_dataset, batch_size, num_workers, hidden_num,
             learning_rate, momentum, times, path_name='result.csv', written=True, optimizer_type='SGD', alpha=0.9, betas=(0.9, 0.99)):
    train_accuracy_list = np.zeros([times])
    validate_accuracy_list = np.zeros([times])
    test_accuracy_list = np.zeros([times])

    for time in range(times):
        print('start to run the time {}'.format(time))

        net = Network(hidden_num)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
        if (optimizer_type == 'RMSprop'):
            optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, alpha=alpha)
        if (optimizer_type == 'Adam'):
            optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=betas)

        train(net, train_dataset, validate_dataset, batch_size, num_workers, criterion, optimizer)
        train_accuracy_list[time] = test(net, train_dataset, criterion) #get the train set final accuracy
        validate_accuracy_list[time] = test(net, validate_dataset, criterion)  # get the train set final accuracy
        test_accuracy_list[time] = test(net, test_dataset, criterion)

    if written:
        file_out = open(path_name, 'w')
        line_1 = 'No,' + ','.join([str(i) for i in range(1, times + 1)])
        line_2 = 'Train Accuracy,' + ','.join([str(train_accuracy) for train_accuracy in train_accuracy_list])
        line_3 = 'Validate Accuracy,' + ','.join(
            [str(validate_accuracy) for validate_accuracy in validate_accuracy_list])
        line_4 = 'Test Accuracy,' + ','.join([str(test_accuracy) for test_accuracy in test_accuracy_list])

        file_out.write(line_1 + '\n')
        file_out.write(line_2 + '\n')
        file_out.write(line_3 + '\n')
        file_out.write(line_4 + '\n')

        print(line_1)
        print(line_2)
        print(line_3)
        print(line_4)

        file_out.close()
    else:
        print('train accuracy: {}'.format(train_accuracy_list))
        print('validate accuracy: {}'.format(validate_accuracy_list))
        print('test accuracy: {}'.format(test_accuracy_list))

    return np.mean(train_accuracy_list), np.mean(validate_accuracy_list), np.mean(test_accuracy_list)