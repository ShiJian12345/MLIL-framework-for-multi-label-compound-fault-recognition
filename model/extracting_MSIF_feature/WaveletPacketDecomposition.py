import numpy as np
import pywt
import pywt.data

def Wavelet_packet_decomposition(array, wavelet='db1', levels=6):

      signal_num, signal_length = array.shape 

      Wavelet_coefficient_matrices = np.zeros((signal_num,int(signal_length/(2**levels)),2**levels)) 
      for i in range(signal_num):
            single_signal = array[i,:] 

            wp = pywt.WaveletPacket(single_signal, wavelet=wavelet, mode='symmetric', maxlevel=levels)

            wavelet_nodeNames_list = [node.path for node in wp.get_level(levels, 'natural')]

            for num, wavelet_nodeName in enumerate(wavelet_nodeNames_list):

                  Wavelet_coefficient_matrices[i][num][:] = wp[wavelet_nodeName].data  

      return Wavelet_coefficient_matrices

def multiple_source_data_WPD(array, wavelet='db1', levels=6):

      signal_num, source_num, signal_length = array.shape
      MSD_WPD = np.zeros((signal_num, source_num, int(signal_length/(2**levels)),2**levels))
      for i in range(source_num):
            MSD_WPD[:, i, :, :] = Wavelet_packet_decomposition(array[:, i, :], wavelet=wavelet, levels=levels)
      return MSD_WPD

if __name__=='__main__':
      x = np.array([i for i in range(4096)]).reshape(1,-1)
      print(x.shape)
      y = Wavelet_packet_decomposition(x)

      print(y)
      print(y.shape)
