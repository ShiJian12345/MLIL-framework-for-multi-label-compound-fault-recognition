
from scipy.io import savemat
import numpy as np
from scipy.io import loadmat
# import cv2

'''
simplt way, only stack the array, is error. the file show the 3 or signal channel of fult data on time domain.

'''


def old_convert_8_unit_gray_signal_channel(array, width=64, height=64):
    '''

    :param array: shape(64*64,) class-ndarray
    :return:
    '''
    array_square = np.square(array)
    min_square = np.min(array_square)
    max_square = np.max(array_square)
    temp = max_square - min_square
    # width = 64l
    # height = 64
    convert_array = np.zeros([width, height])
    for row in range(height):
        for clo in range(width):
            convert_array[row, clo] = 255*(array_square[row*width+clo] - min_square)/temp
    # print(convert_array)
    # print(np.argmax(convert_array))
    # print(np.max(convert_array))
    # print(convert_array.round())
    return convert_array.round()


def convert_8_unit_gray_signal_channel(array, width=64, height=64):
    '''

    :param array: shape(64*64,) class-ndarray
    :return:
    '''
    # array = (array - np.mean(array)) / np.std(array)

    return array
    # pass


# see ae data
def load_mat_func(file_path, channel_list):

    fault_dataset = loadmat(file_path+'.mat')
    dataset=[]
    # expand the data of every channel to 150*4096
    for channel in channel_list:
        dataset.append(fault_dataset[channel])

    # here, save 11 kinds fault data that has 2 channel
    # print(dataset.shape)  # (2, 150, 4096)

    return dataset


def split_train_or_test(one_fault_all_data, one_fault_all_label):
    num_sample = len(one_fault_all_data)  # 150
    num_train = int(num_sample * 160 / 200)

    # index_permutation = np.arange(num_sample)
    # np.random.shuffle(index_permutation)  # 数据打乱
    # train_data = one_fault_all_data[index_permutation][: num_train]
    # test_data = one_fault_all_data[index_permutation][num_train:]
    #
    # train_label = one_fault_all_label[index_permutation][: num_train]
    # test_label = one_fault_all_label[index_permutation][num_train:]

    train_data = one_fault_all_data[: num_train]
    test_data = one_fault_all_data[num_train:]

    train_label = one_fault_all_label[: num_train]
    test_label = one_fault_all_label[num_train:]

    # 4 (120, 2, 4096) (120, 11) (30, 2, 4096) (30, 11)
    print('4',train_data.shape, train_label.shape, test_data.shape, test_label.shape)
    return train_data, train_label, test_data, test_label


def analysis1(folder_path, file_name, channel_list, label_one_hot, channel_num):
    width = 64
    height = 64
    dataset = load_mat_func(folder_path+file_name, channel_list)

    # continue
    # single channel
    if channel_num==1:
        signal_dataset_list = []
        for array, channel in zip(dataset, channel_list):
            print('array',array.shape)  # (150, 4096)
            # according every row, 4096 data points, put these to convert as [0,255]
            # this step is done to normalize the data on every row.
            for row in range(len(array)):
                unit_8_gray_signal_channel = convert_8_unit_gray_signal_channel(array[row,:], width, height)
                # print(unit_8_gray_signal_channel.shape)  #(64*64,)
                signal_dataset_list.append(unit_8_gray_signal_channel)  # [(4096,),...,] 150 ge (4096,), list type
        # convert the list to numpyArray while keep the shape is same
        singnal_channel_data_numpy = np.stack(signal_dataset_list, axis=0)  # (150, 64*64)
        print('singnal_channel_data_numpy',singnal_channel_data_numpy.shape)  # singnal_channel_data_numpy (150, 4096)
        # expand the variable (label_one_hot) to 150 times, for instance: [1 2]*3 == [1 2 1 2 1 2]
        label = np.array(label_one_hot * len(singnal_channel_data_numpy))  # (150*11, )
        signal_channel_label_one_hot = label.reshape(-1, len(label_one_hot))  # (150, 11)
        print('signal_channel_label_one_hot', signal_channel_label_one_hot.shape)  # signal_channel_label_one_hot (150, 11)
        # split the 150 sample into 120 train sample and 30 test sample meanwhile shuffle the rank.
        return split_train_or_test(singnal_channel_data_numpy, signal_channel_label_one_hot)

    # 2 channel
    # according every row, 4096 data points, put these to convert as [0,255]
    three_dataset_list = []
    # from 1 to 150 row, aim to every sample
    for row in range(len(dataset[0])):  # range(150)
        unit_8_gray_pool_signal_channel = []
        # make the 2 kind channel data that every channel has 4096 points to pooled in a list, pay attention to that the row value of every channel is same
        # aim to every channel
        for array in dataset:  # dataset (2,150,4096)
            unit_8_gray_signal_channel = convert_8_unit_gray_signal_channel(array[row,:], width=64, height=64)
            unit_8_gray_pool_signal_channel.append(unit_8_gray_signal_channel)
            # print(unit_8_gray_signal_channel.shape)  #(64*64,)
            # print(unit_8_gray_pool_signal_channel.shape)  #(2,64*64)
        three_dataset_list.append(np.stack(unit_8_gray_pool_signal_channel, axis=0))  # 150 ge (2,64*64)

    three_channel_data_numpy = np.stack(three_dataset_list, axis=0)  # (150, 2, 64*64)
    print('three_channel_data_numpy', three_channel_data_numpy.shape)  # three_channel_data_numpy (150, 2, 4096)
    label = np.array(label_one_hot * len(three_channel_data_numpy))  # (150*11, )
    three_channel_label_one_hot = label.reshape(-1, len(label_one_hot))  # (150, 11)
    print('three_channel_label_one_hot', three_channel_label_one_hot.shape)  # three_channel_label_one_hot (150, 11)

    return split_train_or_test(three_channel_data_numpy, three_channel_label_one_hot)


def pool_data_or_label(signal_channel_data_or_label, data_type):
    return np.concatenate((signal_channel_data_or_label[0][data_type],
                           signal_channel_data_or_label[1][data_type],
                           signal_channel_data_or_label[2][data_type],
                           signal_channel_data_or_label[3][data_type],
                           signal_channel_data_or_label[4][data_type],
                           signal_channel_data_or_label[5][data_type],
                           signal_channel_data_or_label[6][data_type],
                           signal_channel_data_or_label[7][data_type],
                           signal_channel_data_or_label[8][data_type],
                           signal_channel_data_or_label[9][data_type],
                           signal_channel_data_or_label[10][data_type]), axis=0)


def random_shuffle_data(data, label):
    length = len(data)
    shuffle_index = np.arange(length)
    np.random.shuffle(shuffle_index)
    data_shuffle = data[shuffle_index]
    label_shuffle = label[shuffle_index]
    return data_shuffle, label_shuffle


np.set_printoptions(suppress=True)  # console show value as float type

file_folder_path = 'data_enhancement_raw_data/'
channels_list = ['fan_end', 'drive_end']
labels_matrix = np.eye(11)
labels_list = []

for i in range(11):
    labels_list.append(list(labels_matrix[i]))

print(labels_list)  # [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ...]

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
print(file_mat_name_list)  # ['O_P', 'I_P', 'B_P', 'R_P', 'O_I', 'O_B', 'O_R', 'I_B', 'I_R', 'B_R', 'N_P']


data_type = ['train_data_set', 'train_label', 'test_data_set', 'test_label']

signal_channel_all_data_list = []

for file_mat_name, label in zip(file_mat_name_list, labels_list):
    temp = {}
    # (150, 64*64) (150, 11)
    file_mat_name = file_mat_name.replace('_', '-')
    temp[data_type[0]], temp[data_type[1]], temp[data_type[2]], temp[data_type[3]] = analysis1(file_folder_path, file_mat_name, channels_list, label, len(channels_list))
    signal_channel_all_data_list.append(temp)

train_data_raw = pool_data_or_label(signal_channel_all_data_list, data_type[0])  # (11*120,64*64)
train_label_raw = pool_data_or_label(signal_channel_all_data_list, data_type[1])  # (11*120,11)
train_data, train_label = random_shuffle_data(train_data_raw, train_label_raw)

print('train_data',train_data.shape)  # train_data (1320, 2, 4096)
print('train_label',train_label.shape)  # train_label (1320, 11)

test_data_raw = pool_data_or_label(signal_channel_all_data_list, data_type[2])  # (11*120,64*64)
test_label_raw = pool_data_or_label(signal_channel_all_data_list, data_type[3])  # (11*120,11)
test_data, test_label = random_shuffle_data(test_data_raw, test_label_raw)

print('test_data',test_data.shape)  # test_data (330, 2, 4096)
print('test_label',test_label.shape)  # test_label (330, 11)

fault_dataset = {'train_data': train_data,
                 'train_label': train_label,
                 'test_data': test_data,
                 'test_label':test_label}  # 以字典形式保存为 mat 文件


savemat('train_test_dataset-noMeanStd/' + 'HDX-800_compound_fault_11class_fanAndDri-noMeanStd' + '.mat', fault_dataset)


'''
/home/dell/miniconda3/envs/pytorch1.01/bin/python3 
/media/dell/DATA2/opt/FP_NEW/Paper/HDX-800/dataset_HDX-800_compound_fault/save_channel-drive_end_train_test-fanAndDri-channel.py
[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
three_channel_data_numpy (150, 2, 4096)
three_channel_label_one_hot (150, 11)
4 (120, 2, 4096) (120, 11) (30, 2, 4096) (30, 11)
three_channel_data_numpy (150, 2, 4096)
three_channel_label_one_hot (150, 11)
4 (120, 2, 4096) (120, 11) (30, 2, 4096) (30, 11)
three_channel_data_numpy (150, 2, 4096)
three_channel_label_one_hot (150, 11)
4 (120, 2, 4096) (120, 11) (30, 2, 4096) (30, 11)
three_channel_data_numpy (150, 2, 4096)
three_channel_label_one_hot (150, 11)
4 (120, 2, 4096) (120, 11) (30, 2, 4096) (30, 11)
three_channel_data_numpy (150, 2, 4096)
three_channel_label_one_hot (150, 11)
4 (120, 2, 4096) (120, 11) (30, 2, 4096) (30, 11)
three_channel_data_numpy (150, 2, 4096)
three_channel_label_one_hot (150, 11)
4 (120, 2, 4096) (120, 11) (30, 2, 4096) (30, 11)
three_channel_data_numpy (150, 2, 4096)
three_channel_label_one_hot (150, 11)
4 (120, 2, 4096) (120, 11) (30, 2, 4096) (30, 11)
three_channel_data_numpy (150, 2, 4096)
three_channel_label_one_hot (150, 11)
4 (120, 2, 4096) (120, 11) (30, 2, 4096) (30, 11)
three_channel_data_numpy (150, 2, 4096)
three_channel_label_one_hot (150, 11)
4 (120, 2, 4096) (120, 11) (30, 2, 4096) (30, 11)
three_channel_data_numpy (150, 2, 4096)
three_channel_label_one_hot (150, 11)
4 (120, 2, 4096) (120, 11) (30, 2, 4096) (30, 11)
three_channel_data_numpy (150, 2, 4096)
three_channel_label_one_hot (150, 11)
4 (120, 2, 4096) (120, 11) (30, 2, 4096) (30, 11)
train_data (1320, 2, 4096)
train_label (1320, 11)
test_data (330, 2, 4096)
test_label (330, 11)

Process finished with exit code 0

'''