# [Multi-source information fusion deep self-attention reinforcement learning framework for multi-label compound fault recognition]

## Introduction

Aiming at compound fault recognition, multi-label learning easily has a strong comprehension on relevance between simultaneous mechanism faults, such as bearing defect fault and tool wear fault. Moreover, compared with single data source, multiple data sources can more fully monitor the working status of equipment. Consequently, this paper proposes a multi-source information fusion (MSIF) feature to train the multi-label deep reinforcement learning (ML-DRL) model, and develops a multi-source information fusion deep self-attention reinforcement learning (MSIF-DSARL) framework. Firstly, compound fault samples with multiple data sources are transformed into 3D wavelet coefficient tensors. Then the MSIF features are extracted from 3D tensors, using a position self-attention fusion (PSAF) module and a channel self-attention fusion (CSAF) module. Especially, the PSAF module can excavate the internal time-frequency information in every source, and the CSAF module can integrate the information differences between multiple sources. Finally, the ML-DRL model is trained with the MSIF features. In a laboratory experiment and an engineering application, diagnostic results demonstrate powerfully that the proposed framework has better superiority and practicability in recognizing compound fault, than present popular multi-label learning methods.

![image](https://github.com/ShiJian12345/MLIL-framework-for-multi-label-compound-fault-recognition/blob/main/img/8.png)

## Cityscapes testing set result

In the laboratory experiment, the results of nine methods are shown as follow.

![image](./img/tab1.png)

In the engineering application, the results of ten methods are shown as follow.

![image](./img/tab2.png)


## Usage

1. Install pytorch 

   - The code is tested on python3.6 and torch 1.0.1.

2. Dataset
   - Download the [Laboratory experiment](https://pan.baidu.com/s/1fbq3XyqJE6fXTU4neOzvhQ?pwd=2jb6#list/path=%2Fselect_raw_data%2Flaboratory_experiment) dataset and the dataset [Engineering application](https://pan.baidu.com/s/1fbq3XyqJE6fXTU4neOzvhQ?pwd=2jb6#list/path=%2Fselect_raw_data%2Fengineering_application)(password: 2jb6). 
   - Please put dataset in folder `./datasets/select_raw_data`