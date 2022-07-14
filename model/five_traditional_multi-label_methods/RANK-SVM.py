#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from skmultilearn.problem_transform import BinaryRelevance as BRl
from sklearn.svm import SVC
import sklearn.metrics as M
from scipy.io import loadmat
import numpy as np
import sys, os

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



num=2
channel_list = ['fan', 'dri', 'fanAndDri']
channel_selected= channel_list[num]
if not os.path.exists(channel_selected):
    os.makedirs(channel_selected)
sys.stdout = Logger(channel_selected+'/'+channel_selected+"_RANK-SVM_result.txt")

#                0  1  2  3  4  5
#                O  I  B  R  N  P
label_dic = {0: [1, 0, 0, 0, 0, 1],  # 0, 5 OP
             1: [0, 1, 0, 0, 0, 1],  # 1, 5 IP
             2: [0, 0, 1, 0, 0, 1],  # 2, 5 BP
             3: [0, 0, 0, 1, 0, 1],  # 3, 5 RP
             4: [1, 1, 0, 0, 0, 0],  # 0, 1 OI
             5: [1, 0, 1, 0, 0, 0],  # 0, 2 OB
             6: [1, 0, 0, 1, 0, 0],  # 0, 3 OR
             7: [0, 1, 1, 0, 0, 0],  # 1, 2 IB
             8: [0, 1, 0, 1, 0, 0],  # 1, 3 IR
             9: [0, 0, 1, 1, 0, 0],  # 2, 3 BR
             10: [0, 0, 0, 0, 1, 1]}  # 4, 5 NP

mat_file = 'E:\code\seven paper\HDX-800\model\HDX-800_WPD_matrix.mat'
def load_data(path=mat_file):
    f = loadmat(path)
    print(f.keys())  # dict_keys(['__header__', '__version__', '__globals__', 'train_data', 'train_label', 'test_data', 'test_label'])

    x_train, y_train = f['x_WPD_train'], f['train_label']

    x_test, y_test = f['x_WPD_test'], f['test_label']
    # print(type(x_train))  # <class 'numpy.ndarray'>

    print('x_train', x_train.shape)  # x_train (1320, 2, 64, 64)
    print('y_train', y_train.shape)   # y_train (1320, 11)
    print('x_test', x_test.shape)  # x_test (330, 2, 64, 64)
    print('y_test', y_test.shape)  # y_test (330, 11)


    return (x_train, y_train), (x_test, y_test)



def batch_iter(epoch):

    classifier = BRl(classifier=SVC(C=0.6, kernel='linear'))  # 0.62
    classifier.fit(x_train.reshape(1320, -1), Y_train)
    predictions = classifier.predict(x_test.reshape(330, -1))

    acc = M.accuracy_score(Y_test, predictions)
    print("=" * 50)
    print(str(epoch+1))
    print("=" * 50)
    print(acc)
    return acc

def transform_label(araay):
    y_list = []
    for i in range(araay.shape[0]):
        y = araay[i]
        y = np.argmax(y)
        y = label_dic[y]
        y_list.append(y)
    return y_list


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()

    _y_test = transform_label(y_test)
    _y_train = transform_label(y_train)

    Y_test = np.array(_y_test)
    Y_train = np.array(_y_train)
    print(Y_test.shape)  # (330, 6)
    print(Y_train.shape)  # (1320, 6)

    epochs = 10

    accuracy = []
    for epoch in range(epochs):
        accuracy.append(batch_iter(epoch))
    print('=' * 30)
    print('minimum value: ', np.min(accuracy))
    print('average value: ', np.mean(accuracy))
    print('maximum value: ', np.max(accuracy))