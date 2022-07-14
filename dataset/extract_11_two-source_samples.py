

from scipy.io import savemat
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def load_two_sources_sample_func(file_path, file_name):
    two_source_sample = []
    data = loadmat(file_path + file_name + '.mat')
    # print(type(data))  # <class 'dict'>
    data_drive = data['drive_end']  # ndArray : (150, 4096)
    data_fan = data['fan_end']  # ndArray : (150, 4096)
    sample_drive = data_drive[100].flatten()
    sample_fan = data_fan[100].flatten()
    two_source_sample.append(sample_drive)
    two_source_sample.append(sample_fan)
    return two_source_sample


def plot_time_11_fft_image(all_sample, row_num=None, col_num=None):
    '''
    :param all_sample: 11个2源样本数据 （11,2,4096）
    :param row_num: 图片中显示11个样本
    :param col_num: 图片中每个样本有2个源数据
    :return: none，画出11x2的时域图
    '''
    #  three channel time domain image
    fig = plt.figure(figsize=(11, 14), dpi=300)
    row = row_num
    col = col_num
    length = len(all_sample[0][0])
    x_tick = range(length)

    print(length)  # 4096
    x1_tick = [i / samplt_frequence for i in x_tick]
    x1_tick[0] = 0
    print(list(x_tick)[::4*4096])
    print(x1_tick[::4*4096])
    ax = plt.gca()
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    for index in range(row):
        # 画index行的第一个子图
        plt.subplot(row, col, 2*index+1)
        plt.plot(all_sample[index][0], linewidth=0.5)
        # plt.xticks(list(x_tick)[::512]+[4096], x1_tick[::512]+[4096/25600])
        # plt.xticks(list(x_tick)[::4*4096], x1_tick[::4*4096])
        plt.xlim(0, length + 1)
        # plt.ylabel('('+str(index+1)+')', labelpad=-400, rotation=0)

        # 画index行的第二个子图
        plt.subplot(row, col, 2 * index + 2)
        plt.plot(all_sample[index][1], linewidth=0.5)
        plt.xlim(0, length + 1)
    fig.tight_layout(pad=0, w_pad=4, h_pad=2)
    plt.savefig('extract_11_two_sources_samples_time.tif', bbox_inches='tight')
    plt.show()

samplt_frequence = 49152
file_folder_path = 'data_enhancement_raw_data/'

# Four paper uses IN(10 sample), NN(1 sample)

#  ['OP', 'IP', 'BP', 'RP', 'OI', 'OB', 'OR', 'IB', 'IR', 'BR', 'NP']

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

all_fault_sample_list=[]
for file_name in file_mat_name_list:
    file_name = file_name.replace('_', '-')
    all_fault_sample_list.append(load_two_sources_sample_func(file_folder_path, file_name))  # (11, 4096)

plot_time_11_fft_image(all_fault_sample_list, 11, 2)