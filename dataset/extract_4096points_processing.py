

from scipy.io import savemat
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import os

folder = 'processing_data/one_sample/'
if not os.path.exists(folder):
    os.makedirs(folder)

def load_mat_func(file_path, file_name, name_end):
    data = loadmat(file_path+file_name+'.mat')[name_end]
    print(data.shape)  # 11个(150, 4096)
    print(data[100].shape)  # 11个(4096,)
    print(data[100:110].shape)  # 11个(10, 4096)
    print(data[100:110].flatten().shape)  # 11个(40960,)
    one_samples_points = data[100].flatten()
    savemat(folder+'HXD-800_11classes_1sample-'+name_end+'.mat', {'800_11classes_1sample-'+name_end: one_samples_points})
    return one_samples_points


def plot_data_wavelet_graph_wubiankuang(all_sample, name_end):

    length = len(all_sample[0])
    x_tick  = range(length)
    x1_tick = [i/samplt_frequence for i in x_tick]
    x1_tick[0]=0
    plt.close()
    fig = plt.figure(figsize=(10, 2), dpi=300)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(all_sample[0], linewidth=0.5)

    plt.axis('off')
    # plt.xticks([])

    # plt.yticks([])

    # plt.xticks(x_tick[::512], x1_tick[::512])
    # plt.xticks(x_tick[::200*64*64], x2_tick)
    plt.xlim(0,length)
    # plt.ylim(-12,12)
    # plt.ylabel('('+str(index+1)+')', labelpad=-260, rotation=0)
    # plt.ylabel(channels_list[index], rotation=0)
    # if index==2:
        # plt.xlabel('time')
    # if index==0:
    #     plt.title('Raw signal')
    fig.tight_layout(pad=0, w_pad=-1, h_pad=20)

    plt.savefig(folder+'WUBIANKUANG_HDX-800_MSIF-DARL_1sample-'+name_end+'.tif', bbox_inches='tight')

    plt.show()

def plot_data_wavelet_graph_you_biankuang(all_sample, name_end):

    length = len(all_sample[0])
    x_tick  = range(length)
    x1_tick = [i/samplt_frequence for i in x_tick]
    x1_tick[0]=0
    plt.close()
    fig = plt.figure(figsize=(10, 2), dpi=300)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(all_sample[0], linewidth=0.5)

    # plt.axis('off')
    # plt.xticks([])

    # plt.yticks([])

    # plt.xticks(x_tick[::512], x1_tick[::512])
    # plt.xticks(x_tick[::200*64*64], x2_tick)
    plt.xlim(0,length)
    # plt.ylim(-12,12)
    # plt.ylabel('('+str(index+1)+')', labelpad=-260, rotation=0)
    # plt.ylabel(channels_list[index], rotation=0)
    # if index==2:
        # plt.xlabel('time')
    # if index==0:
    #     plt.title('Raw signal')
    fig.tight_layout(pad=0, w_pad=-1, h_pad=20)

    plt.savefig(folder+'YOUBIANKUANG_HDX-800_MSIF-DARL_1sample-'+name_end+'.tif', bbox_inches='tight')
    plt.show()

samplt_frequence = 49152
file_folder_path = 'data_enhancement_raw_data/'

# Four paper uses OI(10 sample), OI(1 sample)

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

def main(name_end):
    all_fault_sample_list=[]
    select_class = 6
    for num,file_name in enumerate(file_mat_name_list):
        if num+1==select_class:
            file_name = file_name.replace('_', '-')
            all_fault_sample_list.append(load_mat_func(file_folder_path, file_name, name_end))

    plot_data_wavelet_graph_wubiankuang(all_fault_sample_list, name_end)
    plot_data_wavelet_graph_you_biankuang(all_fault_sample_list, name_end)

if __name__=='__main__':
    for name_end in ['fan_end', 'drive_end']:
        main(name_end)