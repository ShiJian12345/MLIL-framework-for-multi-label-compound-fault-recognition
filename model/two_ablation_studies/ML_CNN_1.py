import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
from scipy.io import loadmat
import numpy as np
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from TwoDimension_CNN_wuWPD import Discriminator
import sys, os, gc

cuda_select = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_select
# fault_list_label = ['O', 'I', 'B', 'R', 'OI', 'OB', 'OR', 'IB', 'IR', 'BR', 'N']
# 是否使用gpu运算
use_gpu = torch.cuda.is_available()
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

num=1
channel_list = ['fan', 'dri', 'fanAndDri']
channel_selected= channel_list[num]
if not os.path.exists(channel_selected):
    os.makedirs(channel_selected)
sys.stdout = Logger(channel_selected+'/'+'HDX-800_compound_fault_11class_'+channel_selected+"_MLCNN_result.txt")

#                0  1  2  3  4  5
#                O  I  B  R  N  P
label_dic = {0: [1, 0, 0, 0, 0, 1],  # 0, 5 OP
             1: [0, 1, 0, 0, 0, 1],  # 1, 5 IP
             2: [0, 0, 1, 0, 0, 1],  # 2, 5 BP
             3: [0, 0, 0, 1, 0, 1],  # 3, 5 RP
             4: [1, 1, 0, 0, 0, 0],  # 0, 1 OI
             5: [1, 0, 1, 0, 0, 0],  # 0, 2 OB
             6: [1, 0, 0, 1, 0, 0],  # 0, 3 OR
             7: [0, 1, 1, 0, 0, 0],  # 1, 2 IB
             8: [0, 1, 0, 1, 0, 0],  # 1, 3 IR
             9: [0, 0, 1, 1, 0, 0],  # 2, 3 BR
             10: [0, 0, 0, 0, 1, 1]}  # 4, 5 NP

BATCHSIZE = 16
# torch.manual_seed(1000)

mat_file = '/media/dell/DATA2/opt/Seven_paper/HDX-800/model/HDX-800_WPD_matrix.mat'
def load_data(path=mat_file):
    f = loadmat(path)
    print(f.keys())  # dict_keys(['__header__', '__version__', '__globals__', 'train_data', 'train_label', 'test_data', 'test_label'])

    x_train, y_train = f['x_WPD_train'], f['train_label']

    x_test, y_test = f['x_WPD_test'], f['test_label']
    # print(type(x_train))  # <class 'numpy.ndarray'>

    print('x_train', x_train.shape)  # x_train (1320, 2, 64, 64)
    print('y_train', y_train.shape)   # y_train (1320, 11)
    print('x_test', x_test.shape)  # x_test (330, 2, 64, 64)
    print('y_test', y_test.shape)  # y_test (330, 11)


    return (x_train, y_train), (x_test, y_test)

class trainset(Dataset):
    def __init__(self, array_X, array_Y):

        self.loader_x = array_X
        self.loader_y = array_Y

    def __getitem__(self, index):
        x = self.loader_x[index]
        y = self.loader_y[index]
        # x = np.array(x, dtype=bytes)
        # x = x[:,:,:3]
        x = x.astype(np.float32)


        y = np.argmax(y)
        y = label_dic[y]
        y = torch.FloatTensor(y)
        return x[1,:,:].reshape(1, 64, 64), y

    def __len__(self):
        return len(self.loader_x)



(x_train, y_train), (x_test, y_test) = load_data()
# print(type(x_train[0]))
# print(y_train[0])
train_dataset = trainset(x_train, y_train)
train_dataLoader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)

valid_dataset = trainset(x_test, y_test)
valid_dataLoader = DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=True)


dataloaders = {'train': train_dataLoader, 'val': valid_dataLoader}
# 读取数据集大小
dataset_sizes = {'train': train_dataset.__len__(), 'val': valid_dataset.__len__()}




# 训练与验证网络（所有层都参加训练）
def train_model(model, criterion, optimizer, scheduler, folder, num_epochs=25):
    Sigmoid_fun = nn.Sigmoid()
    since = time.time()

    history = []

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        history_epoch_train = []
        history_epoch_test = []


        # 每训练一个epoch，验证一下网络模型
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_precision = 0.0
            running_recall = 0.0
            batch_num = 0

            if phase == 'train':
                # 学习率更新方式
                # scheduler.step()
                #  调用模型训练
                model.train()

                # 依次获取所有图像，参与模型训练或测试
                for data in dataloaders[phase]:
                    # 获取输入
                    inputs, labels = data

                    # print(labels)

                    # 判断是否使用gpu
                    if use_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    # 梯度清零
                    optimizer.zero_grad()

                    # 网络前向运行
                    outputs = model(inputs)
                    # 计算Loss值
                    loss = criterion(Sigmoid_fun(outputs), labels)

                    # 这里根据自己的需求选择模型预测结果准确率的函数
                    # precision, recall = calculate_acuracy_mode_one(Sigmoid_fun(outputs), labels)
                    precision, recall = calculate_acuracy_mode_two(Sigmoid_fun(outputs), labels)

                    # 这里如果计算损失等, 要除以 batchsize=32, 才是平均后的值, 这些都是一个batch批次的
                    # print('loss', loss)
                    # print('precision', precision)
                    # print('recall', recall)


                    running_precision += precision
                    running_recall += recall
                    batch_num += 1
                    # 反传梯度
                    loss.backward()
                    # 更新权重
                    optimizer.step()
                    # 计算一个epoch的loss值和准确率
                    running_loss += loss.item() * inputs.size(0)
            else:
                # 取消验证阶段的梯度
                with torch.no_grad():
                    # 调用模型测试
                    model.eval()
                    # 依次获取所有图像，参与模型训练或测试
                    for data in dataloaders[phase]:
                        # 获取输入
                        inputs, labels = data
                        # 判断是否使用gpu
                        if use_gpu:
                            inputs = inputs.cuda()
                            labels = labels.cuda()

                        # 网络前向运行
                        outputs = model(inputs)
                        # 计算Loss值
                        # BCELoss的输入（1、网络模型的输出必须经过sigmoid；2、标签必须是float类型的tensor）
                        loss = criterion(Sigmoid_fun(outputs), labels)
                        # 计算一个epoch的loss值和准确率
                        running_loss += loss.item() * inputs.size(0)

                        # 这里根据自己的需求选择模型预测结果准确率的函数
                        # precision, recall = calculate_acuracy_mode_one(Sigmoid_fun(outputs), labels)
                        # precision, recall = calculate_acuracy_mode_two(Sigmoid_fun(outputs), labels)
                        precision, recall = calculate_acuracy_mode_two2(Sigmoid_fun(outputs), labels)
                        running_precision += precision
                        running_recall += recall
                        batch_num += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_precision = running_precision / batch_num
            epoch_recall = running_recall / batch_num

            # if epoch<=99:
            if epoch==99:

                # 计算Loss和准确率的均值
            # epoch_loss = running_loss / dataset_sizes[phase]
                print('{} Loss: {:.4f} '.format(phase, epoch_loss))
            # epoch_precision = running_precision / batch_num
                print('{} Precision: {:.4f} '.format(phase, epoch_precision))
            # epoch_recall = running_recall / batch_num
                print('{} Recall: {:.4f} '.format(phase, epoch_recall))
                # torch.save(model.state_dict(), 'models/parameters/The_' + str(epoch) + '_epoch_model_parameters.pkl')
                # torch.save(model, 'models/net/The_' + str(epoch) + '_epoch_model.pkl')
                torch.save(model.state_dict(), folder + '/The_' + str(epoch) + '_epoch_model_parameters.pkl')
                torch.save(model, folder + '/The_' + str(epoch) + '_epoch_model.pkl')


            if phase == 'train':
                history_epoch_train.append([epoch_loss, epoch_precision])

            else:
                history_epoch_test.append([epoch_loss, epoch_precision])

        history.append([history_epoch_train[0][0], history_epoch_train[0][1], history_epoch_test[0][0], history_epoch_test[0][1]])



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return history


def torch_to_numpy(tensor):
    return tensor.data.cpu().numpy()

def torchCPU(tensor):
    return tensor.data.cpu()

def numpy_to_torch(array):
    return torch.tensor(array).float().cuda()


# 计算准确率——方式1
# 设定一个阈值，当预测的概率值大于这个阈值，则认为这幅图像中含有这类标签
def calculate_acuracy_mode_one(model_pred, labels):
    if use_gpu:
        model_pred =torchCPU(model_pred)
        labels =torchCPU(labels)
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    pred_one_num = torch.sum(pred_result)
    if pred_one_num == 0:
        return 0, 0
    target_one_num = torch.sum(labels)
    true_predict_num = torch.sum(pred_result * labels)
    # 模型预测的结果中有多少个是正确的
    precision = true_predict_num / pred_one_num
    # 模型预测正确的结果中，占所有真实标签的数量
    recall = true_predict_num / target_one_num

    return precision.item()/BATCHSIZE, recall.item()/BATCHSIZE


# 计算准确率——方式2
# 取预测概率最大的前top个标签，作为模型的预测结果
def calculate_acuracy_mode_two(model_pred, labels):
    if use_gpu:
        model_pred =torchCPU(model_pred)
        labels =torchCPU(labels)

    # 取前top个预测结果作为模型的预测结果
    precision = 0
    recall = 0
    top = 1
    # print(model_pred, labels)
    # 对预测结果进行按概率值进行降序排列，取概率最大的top个结果作为模型的预测结果
    model_pred_1 = model_pred[:, 0:3]
    model_pred_2 = model_pred[:, 3:7]
    model_pred_3 = model_pred[:, 7:9]

    pred_label_locate_1 = torch.argsort(model_pred_1, descending=True)[:, 0:top]
    pred_label_locate_2 = torch.argsort(model_pred_2, descending=True)[:, 0:top]
    pred_label_locate_2 = pred_label_locate_2 + torch.tensor(3)
    pred_label_locate_3 = torch.argsort(model_pred_3, descending=True)[:, 0:top]
    pred_label_locate_3 = pred_label_locate_3 + torch.tensor(7)
    pred_label_locate = torch.cat((pred_label_locate_1, pred_label_locate_2, pred_label_locate_3), dim=1)
    # print(pred_label_locate)
    for i in range(model_pred.shape[0]):
        temp_label = torch.zeros(1, model_pred.shape[1])
        temp_label[0, pred_label_locate[i]] = 1
        target_one_num = torch.sum(labels[i])
        true_predict_num = torch.sum(temp_label * labels[i])
        # 对每一幅图像进行预测准确率的计算
        precision += true_predict_num / (3*top)
        # 这下面是对每一个样本的预测平均准确度的判断
        # print(true_predict_num / top)
        # 对每一幅图像进行预测查全率的计算
        recall += true_predict_num / target_one_num
    return precision/BATCHSIZE, recall/BATCHSIZE



def calculate_acuracy_mode_two2(model_pred, labels):
    if use_gpu:
        model_pred =torchCPU(model_pred)
        labels =torchCPU(labels)

    # 取前top个预测结果作为模型的预测结果
    precision = 0
    recall = 0
    top = 2
    # 对预测结果进行按概率值进行降序排列，取概率最大的top个结果作为模型的预测结果
    pred_label_locate = torch.argsort(model_pred, descending=True)[:, 0:top]

    for i in range(model_pred.shape[0]):
        temp_label = torch.zeros(1, model_pred.shape[1])
        temp_label[0, pred_label_locate[i]] = 1
        target_one_num = torch.sum(labels[i])
        true_predict_num = torch.sum(temp_label * labels[i])
        # 对每一幅图像进行预测准确率的计算
        precision += true_predict_num / top
        # 这下面是对每一个样本的预测平均准确度的判断
        # print(true_predict_num / top)
        # 对每一幅图像进行预测查全率的计算
        recall += true_predict_num / target_one_num
    return precision/BATCHSIZE, recall/BATCHSIZE



# 精调AlexNet
def batch_iter(epoch, type_file):
    dataset = 'MLCNN_HDX-800_compound_fault_11class'
    print("=" * 50)
    print(str(epoch+1))
    print("=" * 50)
    folder = type_file + '/' + str(epoch+1)
    if not os.path.exists(folder):
        os.makedirs(folder)


    cnn = Discriminator(input_dim=64, output_dim=6)
    device = torch.device("cuda:"+'0' if torch.cuda.is_available() else "cpu")
    cnn = cnn.to(device)

    # print(resnet50)

    # 如果最后一层加了softmax函数，就用nn.NLLLoss(),
    # 如果用CrossEntropyLoss()的时候，网络的最后输出不要加激活
    # loss_func = nn.NLLLoss()

    # 多标签分类使用nn.BCELoss()
    loss_func = nn.BCELoss()

    optimizer = optim.Adam(cnn.parameters(), 5e-5)  # 0.001
    # 定义学习率的更新方式，每5个epoch修改一次学习
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    history = train_model(cnn, loss_func, optimizer, exp_lr_scheduler, folder, num_epochs=100)


    history = np.array(history)
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig(folder + '/' + dataset + '_loss_curve.png')
    # plt.show()

    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(folder + '/'+dataset + '_accuracy_curve.png')
    # plt.show()
    acc = history[99][3]
    del cnn, history, optimizer, exp_lr_scheduler, loss_func
    gc.collect()
    return acc

if __name__ == '__main__':
    epochs = 20
    type_file = channel_selected+'/'+'figures_MLCNN_save'
    accuracy = []
    for epoch in range(epochs):
    # for epoch in [i for i in range(18)]+[19]:
    # for epoch in [12]:
        accuracy.append(batch_iter(epoch, type_file))
    print('=' * 30)
    print('minimum value: ', np.min(accuracy))
    print('average value: ', np.mean(accuracy))
    print('maximum value: ', np.max(accuracy))
