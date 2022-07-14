
import os
from scipy.io import savemat
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
# import cv2
import pandas as pd

NUM_TRAIN = 120
NUM_TEST = 30

shufftle_arange_train = np.arange(NUM_TRAIN)
np.random.shuffle(shufftle_arange_train)
shufftle_arange_test=np.arange(NUM_TEST)
np.random.shuffle(shufftle_arange_test)
shufftle_arrange_train_test=np.concatenate((shufftle_arange_train, shufftle_arange_test+NUM_TRAIN))



def data_enhance(array, length, multip):
    """
    数据增强，使用长度为length的窗口进行滑动，选取数据，数据之间有重叠
    使数据量变为原理的multip倍

    Parameters
    ----------
    array  : array 例如 (?, 1)
    length : 截取数据的窗口长度，64*64
    multip ：数据增强的倍数multiple，NUM_TRAIN+NUM_TEST

    Returns
    ----------
    array_enhance.T : 增强以后的数据  (NUM_TRAIN+NUM_TEST,64*64)
    """
    print(" array.shape", array.shape)  # ((244000, 1)
    array_enhance = np.empty([length , multip])  # (4096,150)
    array_len = len(array) - length
    overlap = int(array_len/(multip - 1))  # 窗口滑移时的重叠长度
    print("overlap: ", overlap)  # 1610
    for i in range(int(multip/2)):
        array_enhance[:, i] = array[(overlap * i) : (overlap * i + length),0]  # 从前往后写入数据
        # print("array[(overlap * i) : (overlap * i + length)].shape", array[(overlap * i) : (overlap * i + length)].shape)  # 2048
        # print("array_enhance[:, i].shape", array_enhance[:, i].shape)
        array_enhance[:, multip -i -1] = array[(array_len - overlap * i): (array_len - overlap * i + length),0]  # 从后往前写入数据
    if multip % 2 == 1:
        array_enhance[:, int(multip / 2)] = array[int(array_len / 2) : int(array_len / 2 + length),0]  # 如果multip是奇数则中间再插补一个2048的数据
    print("array_enhance.T.shape",array_enhance.T.shape)  # (150, 4096)
    return array_enhance.T

# array = np.array(range(64)).reshape([8,8])
# arrayb = np.array(range(64)).reshape((64,-1))
# print('='*50)
# print(arrayb)
# print('='*50)
# array_enhance = data_enhance(arrayb, 32,4)
# print(array_enhance)
# print('='*50)
# print(array_enhance.shape)


'''
simplt way, only stack the array, is error. the file show the 3 or signal channel of fult data on time domain.

'''


def plot_time(array, file_name, type_list):

    plt.subplot(2,1,1)
    plt.plot(array[0])
    plt.title(file_name + type_list[0])

    plt.subplot(2,1,2)
    plt.plot(array[1])
    plt.title(file_name + type_list[1])

    plt.show()



# see ae data
def load_mat_func(file_path, file_name, channel_list, split_item, shufftle_arange):
    '''
    If data contains three channels:
        return:  (3, NUM_TRAIN+NUM_TEST, 64*64)
    elif data contains one channel:
        return:  (NUM_TRAIN+NUM_TEST, 64*64)
    '''
    channel_list_num = [0, 1]
    # usecols indicates which column is readed. (0,1,...)
    # column starts counting from 0.
    # here, second channel (fan-end) is selected while setting usecols[0]
    data_fanEnd = np.array(pd.read_csv(file_path+file_name+'.csv', usecols=[channel_list_num[0]]))
    # here, second channel (drive-end) is selected while setting usecols[1]
    data_driveEnd = np.array(pd.read_csv(file_path+file_name+'.csv', usecols=[channel_list_num[1]]))

    # 1 every csv file contains two channel data
    # 2 in every channel , the shape of data_driveEnd is (245764, 1),
    # 3 so the shape of data_driveEnd[:,0] is (245764,)

    # expand the data of channel fan_end to (NUM_TRAIN+NUM_TEST)*4096
    start = split_item[0]
    end = split_item[1]

    data_fan_enhance = split_train_and_test_data_enhancement_respectively(data_fanEnd[start:end], 64*64)
    # data_fan_enhance = data_enhance(data_fanEnd[split_item[0]:split_item[1]], 64*64, NUM_TRAIN+NUM_TEST)
    data_fan_enhance_shufftle = data_fan_enhance[shufftle_arange]

    # expand the data of channel dri_end to (NUM_TRAIN+NUM_TEST)*4096
    data_dri_enhance = split_train_and_test_data_enhancement_respectively(data_driveEnd[start:end], 64 * 64)
    # data_dri_enhance = data_enhance(data_driveEnd[split_item[0]:split_item[1]], 64*64, NUM_TRAIN+NUM_TEST)
    data_dri_enhance_shufftle = data_dri_enhance[shufftle_arange]


    fault_dataset = {channel_list[0]: data_fan_enhance_shufftle,
                     channel_list[1]: data_dri_enhance_shufftle}  # 以字典形式保存为 mat 文件
    # here, save (number of bearings * number of tools) kinds of fault data that has 3 channel

    print(data_dri_enhance_shufftle.shape)  # (150, 4096)

    file_name = file_name.replace('_', '-')
    savemat('data_enhancement_raw_data/'+file_name+'.mat', fault_dataset)

def split_train_and_test_data_enhancement_respectively(array_total, sample_length):
    '''
    1 this function first split raw signal to train signal and test signal,
    2 and then enhance them, (NUM_TRAIN, 4096), (NUM_TEST, 4096)
    3 finally, concate two enhancement result
    Parameters
    ----------
    array_total: split signals
    sample_length: 4096
    total_size_dataset: NUM_TRAIN+NUM_TEST

    Returns: (NUM_TRAIN+NUM_TEST, 4096)
    -------
    '''

    signal_length_total = len(array_total)  # end-start=split_item[1]-split_item[0]=245000-1000
    # find how many points is train signal
    point_train = int(signal_length_total * NUM_TRAIN / (NUM_TRAIN+NUM_TEST))
    # split raw signal on point_train and get two sub-signal
    split_train_signal = array_total[:point_train]
    split_test_signal = array_total[point_train:]
    # data_enhance train signal and test signal respectively
    data_train_enhance = data_enhance(split_train_signal, sample_length, NUM_TRAIN)  # (NUM_TRAIN, 4096) numpyArray
    data_test_enhance = data_enhance(split_test_signal, sample_length, NUM_TEST)  # (NUM_TEST, 4096) numpyArray
    # finally, combine the enhancement result to obtain (NUM_TRAIN+NUM_TEST, 4096)
    data_train_and_test_enhance = np.concatenate((data_train_enhance, data_test_enhance), axis=0)
    return data_train_and_test_enhance  # (NUM_TRAIN+NUM_TEST, 4096)


def analysis1(folderpath, file_name, channel_list, split_item, shufftle_arange):
    load_mat_func(folderpath, file_name, channel_list, split_item, shufftle_arange)


np.set_printoptions(suppress=True)  # console show value as float type

labels_matrix = np.eye(11)

folderpath = 'select_raw_data/'
# In 'channel_list' variable, value 0 indicates the channel 'fan-end', and 1 indicates the channel 'drive-end'.
channel_list = ['fan_end', 'drive_end']

file_mat_name_list = ['O_P',
                      'I_P',
                      'B_P',
                      'R_P',
                      'O_I',
                      'O_B',
                      'O_R',
                      'I_B',
                      'I_R',
                      'B_R',
                      'N_P']
print(file_mat_name_list)

# fault position of broken bearing is drive-end with 2850 speed, and fourth paper only selects channel driveEnd.
split_list = [[1000, 245000],  # O_P
              [1000, 245000],  # I_P
              [1000, 245000],  # B_P
              [1000, 245000],  # R_P
              [1000, 245000],  # O_I
              [1000, 245000],  # O_B
              [1000, 245000],  # O_R
              [1000, 245000],  # I_B
              [1000, 245000],  # I_R
              [1000, 245000],  # B_R
              [1000, 245000]   # N_P
             ]

for file_name, split in zip(file_mat_name_list, split_list):

    analysis1(folderpath, file_name, channel_list, split, shufftle_arrange_train_test)
