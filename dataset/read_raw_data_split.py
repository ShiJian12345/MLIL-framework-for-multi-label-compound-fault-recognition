from scipy.io import savemat
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_mat_func(file_path, file_name, channel_list):
    channel_list_num = [0, 1]
    # usecols indicates which column is readed. (0,1,...)
    # column starts counting from 0.
    # here, second channel (fan-end) is selected while setting usecols[0]
    data_fanEnd = np.array(pd.read_csv(file_path+'.csv', usecols=[channel_list_num[0]]))
    # here, second channel (drive-end) is selected while setting usecols[1]
    data_driveEnd = np.array(pd.read_csv(file_path+'.csv', usecols=[channel_list_num[1]]))

    # 1 every csv file contains two channel data
    # 2 in every channel , the shape of data_driveEnd is (245764, 1),
    # 3 so the shape of data_driveEnd[:,0] is (245764,)
    data_dri = data_driveEnd[:, 0]
    data_fan = data_fanEnd[:, 0]
    print('='*30)
    print(file_name)
    print(len(data_fan), len(data_dri))

    plot_time((data_fan[1000:245000], data_dri[1000:245000]), file_name, channel_list)


def plot_time(array, file_name, channel_list):
    # the input array must be one-dimensional data

    plt.subplot(2,1,1)
    plt.plot(array[0])
    plt.title(file_name + '-' + channel_list[0])

    plt.subplot(2,1,2)
    plt.plot(array[1])
    plt.title(file_name + '-' + channel_list[1])

    plt.show()


samplt_frequence = 49152
file_folder_path = 'select_raw_data/'


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

# In 'channel_list' variable, value 0 indicates the channel 'fan-end', and 1 indicates the channel 'drive-end'.
channel_list = ['fan_end', 'drive_end']

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


for file_name in file_mat_name_list:
    load_mat_func(file_folder_path+file_name, file_name, channel_list)


