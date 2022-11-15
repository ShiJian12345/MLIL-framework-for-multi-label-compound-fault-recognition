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


def main_2D_WPD_matrix(file_signals_path):
    (x_trains, y_trains), (x_tests, y_tests) = load_data(file_signals_path)

    x_wavelet_coefficient_matrices_train = multiple_source_data_WPD(x_trains)
    x_wavelet_coefficient_matrices_test = multiple_source_data_WPD(x_tests)
    print('x_wavelet_coefficient_matrices_train:', x_wavelet_coefficient_matrices_train.shape)
    print('x_wavelet_coefficient_matrices_test:', x_wavelet_coefficient_matrices_test.shape)

    fault_dataset = {'x_WPD_train': x_wavelet_coefficient_matrices_train,
                     'x_WPD_test': x_wavelet_coefficient_matrices_test,
                     'train_label': y_trains,
                     'test_label': y_tests}  

    savemat('HDX-800_WPD_matrix' + '.mat', fault_dataset)


if __name__ == "__main__":
    file_signals_path = './raw_signals_HDX-800/HDX-800_compound_fault_11class_fanAndDri-noMeanStd.mat'
    main_2D_WPD_matrix(file_signals_path)
    print("exit")

