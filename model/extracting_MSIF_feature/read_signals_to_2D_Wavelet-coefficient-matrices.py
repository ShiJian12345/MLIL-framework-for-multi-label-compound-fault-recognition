import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
from PIL import Image
import gc
from WaveletPacketDecomposition import multiple_source_data_WPD

def load_data(path):
    f = loadmat(path)

    x_train, y_train = f['train_data'], f['train_label']

    x_test, y_test = f['test_data'], f['test_label']
    print(type(x_train))  # <class 'numpy.ndarray'>
    print('signals_x_train:', x_train.shape)
    print('signals_x_test:', x_test.shape)
    print('signals_y_train:', y_train.shape)
    print('signals_y_test:', y_test.shape)
    return (x_train, y_train), (x_test, y_test)
'''
该数据共有11类，每个类别有120个训练样本，一共有1320个训练样本，有30个测试样本，一共有330个测试样本
(1320, 2, 4096)
(330, 2, 4096)
y_train(1320, 11)
y_test(330, 11)
'''

def main_2D_WPD_matrix(file_signals_path):
    (x_trains, y_trains), (x_tests, y_tests) = load_data(file_signals_path)

    x_wavelet_coefficient_matrices_train = multiple_source_data_WPD(x_trains)
    x_wavelet_coefficient_matrices_test = multiple_source_data_WPD(x_tests)
    print('x_wavelet_coefficient_matrices_train:', x_wavelet_coefficient_matrices_train.shape)
    print('x_wavelet_coefficient_matrices_test:', x_wavelet_coefficient_matrices_test.shape)

    fault_dataset = {'x_WPD_train': x_wavelet_coefficient_matrices_train,
                     'x_WPD_test': x_wavelet_coefficient_matrices_test,
                     'train_label': y_trains,
                     'test_label': y_tests}  # 以字典形式保存为 mat 文件

    savemat('HDX-800_WPD_matrix' + '.mat', fault_dataset)


if __name__ == "__main__":
    file_signals_path = './raw_signals_HDX-800/HDX-800_compound_fault_11class_fanAndDri-noMeanStd.mat'
    main_2D_WPD_matrix(file_signals_path)
    print("exit")


'''
<class 'numpy.ndarray'>
signals_x_train: (1320, 2, 4096)
signals_x_test: (330, 2, 4096)
signals_y_train: (1320, 11)
signals_y_test: (330, 11)
x_wavelet_coefficient_matrices_train: (1320, 2, 64, 64)
x_wavelet_coefficient_matrices_test: (330, 2, 64, 64)
exit
'''