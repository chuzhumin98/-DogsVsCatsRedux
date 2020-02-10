from DataLoad import DataLoad
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import os
import pickle as pkl

def load_object(path):
    with open(path,'rb') as f:
        obj = pkl.load(f)
    return obj

def save_object(obj,path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)

if __name__ == '__main__':
    mode = 2 # 0: original, 1: resnet embedding, 2: VGG embedding
    method = 0  # 0: LR, 1: SVM, 2: KNN, 3: DT, 4: RF
    run_name = 'vgg'

    print('run with mode={}, method={}, run_name={}'.format(mode, method, run_name))

    if mode == 0:
        path_data = './cache_fc_full'
        loaded = True
        if not loaded:
            loader = DataLoad(sampled=False, model_type='baseline')
            images_train, labels_train, images_val, labels_val, images_test, test_filenames = loader.get_data(flat=True)
            if not os.path.exists(path_data):
                os.makedirs(path_data)

            save_object(images_train, os.path.join(path_data, 'images_train.pkl'))
            save_object(images_val, os.path.join(path_data, 'images_val.pkl'))
            save_object(images_test, os.path.join(path_data, 'images_test.pkl'))
            save_object(labels_train, os.path.join(path_data, 'labels_train.pkl'))
            save_object(labels_val, os.path.join(path_data, 'labels_val.pkl'))
            save_object(test_filenames, os.path.join(path_data, 'test_filenames.pkl'))
        else:
            images_train = load_object(os.path.join(path_data, 'images_train.pkl'))
            images_val = load_object(os.path.join(path_data, 'images_val.pkl'))
            images_test = load_object(os.path.join(path_data, 'images_test.pkl'))
            labels_train = load_object(os.path.join(path_data, 'labels_train.pkl'))
            labels_val = load_object(os.path.join(path_data, 'labels_val.pkl'))
            test_filenames = load_object(os.path.join(path_data, 'test_filenames.pkl'))
    else:
        if mode == 1:
            path_embedding = 'cache/embedding_resnet.pkl'
        else:
            path_embedding = 'cache/embedding_vgg_nonshuffle.pkl'
        data = load_object(path_embedding)
        images_train, labels_train = data['train']
        images_val, labels_val = data['val']
        images_test, test_filenames = data['test']

    print(images_train.shape)


    if method == 0:
        logistic = LogisticRegression()
        clf = logistic.fit(images_train, labels_train)
    elif method == 1:
        svm_model = SVC(kernel='rbf')
        clf = svm_model.fit(images_train, labels_train)
    elif method == 2:
        kneighbor = KNeighborsClassifier()
        clf = kneighbor.fit(images_train, labels_train)
    elif method == 3:
        decision_tree = DecisionTreeClassifier(criterion='gini')
        clf = decision_tree.fit(images_train, labels_train)
    elif method == 4:
        random_forest = RandomForestClassifier(n_estimators=20, criterion='gini')
        clf = random_forest.fit(images_train, labels_train)

    if not os.path.exists('./model'):
        os.makedirs('./model')


    if method == 0:
        save_object(clf, './model/logistic_{}.pkl'.format(run_name))
    elif method == 1:
        save_object(clf, './model/svm_{}.pkl'.format(run_name))
    elif method == 2:
        save_object(clf, './model/kneighbor_{}.pkl'.format(run_name))
    elif method == 3:
        save_object(clf, './model/DT_{}.pkl'.format(run_name))
    elif method == 4:
        save_object(clf, './model/RF_{}.pkl'.format(run_name))

    pred_val = clf.predict(images_val)
    print(pred_val)
    print(float(np.sum(pred_val == labels_val)) / float(len(labels_val)))
    print(float(np.sum(pred_val == labels_val)))

    pred_test = clf.predict(images_test)

    prediction_dict = {}
    for i in range(len(test_filenames)):
        id = int(test_filenames[i].split('.', 1)[0])
        prediction_dict[id] = pred_test[i]

    if not os.path.exists('result'):
        os.makedirs('result')

    if method == 0:
        file_out = open('./result/result_Logistic_{}.csv'.format(run_name), 'w')
    elif method == 1:
        file_out = open('./result/result_svm_{}.csv'.format(run_name), 'w')
    elif method == 2:
        file_out = open('./result/result_kneighbor_{}.csv'.format(run_name), 'w')
    elif method == 3:
        file_out = open('./result/result_DT_{}.csv'.format(run_name), 'w')
    elif method == 4:
        file_out = open('./result/result_RF_{}.csv'.format(run_name), 'w')

    file_out.write('id,label\n')
    for key in sorted(prediction_dict.keys()):
        file_out.write('{},{}\n'.format(key, prediction_dict[key]))

    file_out.close()