# [Multi-source information fusion deep self-attention reinforcement learning framework for multi-label compound fault recognition]

## Introduction

Aiming at compound fault recognition, multi-label learning easily has a strong comprehension on relevance between simultaneous mechanism faults, such as bearing defect fault and tool wear fault. Moreover, compared with single data source, multiple data sources can more fully monitor the working status of equipment. Consequently, this paper proposes a multi-source information fusion (MSIF) feature to train the multi-label deep reinforcement learning (ML-DRL) model, and develops a multi-source information fusion deep self-attention reinforcement learning (MSIF-DSARL) framework. Firstly, compound fault samples with multiple data sources are transformed into 3D wavelet coefficient tensors. Then the MSIF features are extracted from 3D tensors, using a position self-attention fusion (PSAF) module and a channel self-attention fusion (CSAF) module. Especially, the PSAF module can excavate the internal time-frequency information in every source, and the CSAF module can integrate the information differences between multiple sources. Finally, the ML-DRL model is trained with the MSIF features. In a laboratory experiment and an engineering application, diagnostic results demonstrate powerfully that the proposed framework has better superiority and practicability in recognizing compound fault, than present popular multi-label learning methods.

![image](./img/8.jpg)

## Cityscapes testing set result

In the laboratory experiment, the results of nine methods are shown as follow.

![image](./img/table1.jpg)

In the engineering application, the results of ten methods are shown as follow.

![image](./img/table2.jpg)


## Usage

1. Install pytorch 

   - The code is tested on python3.6 and torch 1.0.1.

2. Dataset
   - Download the [Laboratory experiment](https://pan.baidu.com/s/1fbq3XyqJE6fXTU4neOzvhQ?pwd=2jb6#list/path=%2Fselect_raw_data%2Flaboratory_experiment) dataset and the dataset [Engineering application](https://pan.baidu.com/s/1fbq3XyqJE6fXTU4neOzvhQ?pwd=2jb6#list/path=%2Fselect_raw_data%2Fengineering_application). 
   - Please put dataset in folder `./datasets/select_raw_data`

3. Evaluation for DANet

   - Download trained model [DANet101](https://drive.google.com/open?id=1XmpFEF-tbPH0Rmv4eKRxYJngr3pTbj6p) and put it in folder `./experiments/segmentation/models/`

   - `cd ./experiments/segmentation/`

   - For single scale testing, please run:

   - ```shell
     CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset citys --model danet --backbone resnet101 --resume  models/DANet101.pth.tar --eval --base-size 2048 --crop-size 768 --workers 1 --multi-grid --multi-dilation 4 8 16 --os 8 --aux --no-deepstem
     ```

   - Evaluation Result

     The expected scores will show as follows: DANet101 on cityscapes val set (mIoU/pAcc): **79.93/95.97**(ss) 

4. Evaluation for DRANet

   - Download trained model [DRANet101](https://drive.google.com/file/d/1xCl2N0b0rVFH4y30HCGfy7RY3-ars7Ce/view?usp=sharing) and put it in folder `./experiments/segmentation/models/`

   - Evaluation code is in folder `./experiments/segmentation/`

   - `cd ./experiments/segmentation/`

   - For single scale testing, please run:

   - ```shell
     CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset citys --model dran --backbone resnet101 --resume  models/dran101.pth.tar --eval --base-size 2048 --crop-size 768 --workers 1 --multi-grid --multi-dilation 4 8 16 --os 8 --aux
     ```

   - Evaluation Result

     The expected scores will show as follows: DRANet101 on cityscapes val set (mIoU/pAcc): **81.63/96.62** (ss) 

