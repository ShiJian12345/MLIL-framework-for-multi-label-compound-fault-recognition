import numpy as np
import pywt
import pywt.data

#获取样本矩阵的特征向量
def Wavelet_packet_decomposition(array, wavelet='db1', levels=6):
      '''
      该函数将输入进来的原始信号样本转变成小波系数矩阵
      :param array: 2D的ndAarray类型的信号样本数据，（样本数量，每个样本信号的长度） 例子：（120，4096）
      :param wavelet: 选取小波基的类型  例子： ‘db1’
      :param levels: 小波包分解的层数  例子： 6
      :return: 3D的ndArray类型的小波系数矩阵 （样本数量，每个信号样本的长度/2的levels次方，2的levels次方） 例子：（120，64，64）
      '''
      signal_num, signal_length = array.shape #获取矩阵的列数和行数，即样本维数 100 * 4096
      # 这里首先定义一个小波系数空矩阵，第一维度是样本数目，第二维度是分解后子频带的数目，第三维度是每个子频带中信号的长度
      Wavelet_coefficient_matrices = np.zeros((signal_num,int(signal_length/(2**levels)),2**levels)) #定义样本特征向量 #Array 形式
      for i in range(signal_num):
            single_signal = array[i,:] #对第i个信号样本做小波包分解
            #进行小波变换，提取样本特征
            wp = pywt.WaveletPacket(single_signal, wavelet=wavelet, mode='symmetric', maxlevel=levels) #小波包三层分解
            # print([node.path for node in wp.get_level(6, 'natural')])   #第6层有64个
            wavelet_nodeNames_list = [node.path for node in wp.get_level(levels, 'natural')]
            # print(wavelet_nodeNames_list)
            #获取最后一层的节点系数
            for num, wavelet_nodeName in enumerate(wavelet_nodeNames_list):
                  #最后一层所有节点按照子频带从小到大的顺序组合成特征向量
                  Wavelet_coefficient_matrices[i][num][:] = wp[wavelet_nodeName].data  #Array 形式

      return Wavelet_coefficient_matrices

def multiple_source_data_WPD(array, wavelet='db1', levels=6):
      '''
      从多个通道的数据提取小波系数矩阵，便于后续的通道融合
      :param array: 3D多个通道信号样本 （信号样本数量，通道数量，每个信号样本长度）  例子：（120，3，4096）
      :return: 4D小波系数矩阵  （信号样本数量，通道数量，每个信号样本的长度/2的levels次方，2的levels次方）  例子：（120，3，64，64）
      '''
      signal_num, source_num, signal_length = array.shape
      MSD_WPD = np.zeros((signal_num, source_num, int(signal_length/(2**levels)),2**levels))
      for i in range(source_num):
            MSD_WPD[:, i, :, :] = Wavelet_packet_decomposition(array[:, i, :], wavelet=wavelet, levels=levels)
      return MSD_WPD

if __name__=='__main__':
      x = np.array([i for i in range(4096)]).reshape(1,-1)
      print(x.shape)
      y = Wavelet_packet_decomposition(x)
      # y = multiple_source_data_WPD(x.reshape(1,1,-1))
      print(y)
      print(y.shape)
