from utils import plot_with_labels
import numpy as np
from mnist_env import MNISTEnv
from actor_critic_agent import ActorCriticNNAgent
from torchvision import models
import torch.nn as nn
import torch, os
import torch.nn.functional as F
from scipy.io import loadmat, savemat
from itertools import count
from TwoDimension_CNN import Discriminator
# 该文件可以保存tsne转化后的矩阵结果以及3个时间步时每个时间步的2维和3维图, 并测试模型准确率

device = torch.device("cuda:0" if not torch.cuda.is_available() else "cpu")


def load_data(path):
    f = loadmat(path)
    print(f.keys())  # dict_keys(['__header__', '__version__', '__globals__', 'train_data', 'train_label', 'test_data', 'test_label'])
    if 'WPD' not in path:
        x_train, y_train = f['train_data'], f['train_label']

        x_test, y_test = f['test_data'], f['test_label']
    else:
        x_train, y_train = f['x_WPD_train'], f['train_label']

        x_test, y_test = f['x_WPD_test'], f['test_label']

    # print(type(x_train))  # <class 'numpy.ndarray'>

    print('x_train', x_train.shape)  # x_train (1760, 4096)
    print('y_train', y_train.shape)   # y_train (1760, 11)
    print('x_test', x_test.shape)  # x_test (440, 4096)
    print('y_test', y_test.shape)  # y_test (440, 11)


    return (x_train, y_train), (x_test, y_test)


def convert_to_one_hot(y, C):
    return np.eye(C)[y]


def get_action_history(array_initial, array_actions, num):
    temp = np.zeros((len(array_actions), 1))
    if num == 1:
        for i, action in enumerate(array_actions):
            # array_initial[i] = convert_to_one_hot(action[-1], 6)
            temp[i] = np.array(action[-2])

    return temp


class Actor(nn.Module):
    def __init__(self, input_dim=64, output_dim=6):
        self.output_dim = output_dim
        super(Actor, self).__init__()
        self.input_dim = input_dim
        # resnet50 = models.resnet18(pretrained=False)
        # resnet50.fc = nn.Sequential()
        # self.resnet = resnet50
        cnn = Discriminator(output_dim=6)
        cnn.fc = nn.Sequential()
        self.cnn = cnn

        self.lin0 = nn.Linear(512+1, 64)

        self.out_digit = nn.Linear(64, 6)

    def forward(self, x, y):

        x = x.view(-1, 2, self.input_dim, self.input_dim)  # (batch_size, 1, 4096, 1)
        # print(x.size())
        y = y.view(-1,1)
        # x4 = self.resnet(x)
        x4, MSIF = self.cnn(x, tsne=True)

        x4 = torch.cat((x4, y), 1)

        x_ = self.lin0(x4)
        x_ = F.relu(x_)

        x = self.out_digit(x_)
        action_prob = F.softmax(x, dim=1)

        return action_prob, x, x4[:, :512], x_, MSIF.view(-1, 8*self.input_dim*self.input_dim)


def obtained_samples_actions(path):
    num=100
    path_actor = path + 'net_parameters/' + str(num) + '_actor.pkl'
    path_critic = path + 'net_parameters/' + str(num) + '_critic.pkl'

    actor_net = torch.load(path_actor, map_location='cpu')
    critic_net = torch.load(path_critic, map_location='cpu')

    agent = ActorCriticNNAgent(Acotr_Net=actor_net, Critic_Net=critic_net)

    env = MNISTEnv(type='test', seed=None)

    # print(len(env.X))

    # 获取两个历史动作

    samples_action = []
    reward_all = []

    for i in range(len(env.X)):
        rewards = []
        # 该处重置当前环境，也就是样本
        state = env.reset(i)

        for t in count():
            action, action_prob, loss_reward = agent.select_action(state, env)
            # print(action, action_prob)  # 8 0.35509148240089417

            next_state, reward, done, _ = env.step(action, loss_reward)
            state = next_state
            rewards.append(reward)
            if done:
                break

        reward_all.append(rewards)

        samples_action.append(env.B)

    print("test_accuracy: %.3f" % (np.mean(reward_all)))
    return samples_action


def main(model_path, raw_mat_file_path_1D_WPD, raw_mat_file_path_2D_WPD, NUM_TSNE, current):
    # calculate test average reward
    print("=" * 50)
    print("Testing...")

    # initialize agent
    num = 100
    path_actor = model_path + 'net_parameters/' + str(num) + '_actor_parameters.pkl'

    # 选择保存tsne图片的地址
    if current==False:
        # 下面代码是将变换的tsne图片保存到results中
        path_tsne = model_path + 'tsne_files/'
    else:
        # 下面代码是将变换的tsne图片保存到当前目录下
        path_tsne = 'TSNE/' + str(NUM_TSNE) + '/tsne_files/'


    if not os.path.exists(path_tsne):
        os.makedirs(path_tsne+'tsne_mats')
        os.makedirs(path_tsne+'tsne_pictures')

    net = Actor()
    net.load_state_dict(torch.load(path_actor, map_location='cpu'))
    net.eval()
    net.cnn.eval()
    # print(net)
    net = net.to(device)
    # 分别读取1D原始信号和2D小波包系数矩阵
    # 下面是1D
    (x_train_1D, y_train_1D), (x_test_1D, y_test_1D) = load_data(raw_mat_file_path_1D_WPD)
    #下面是2D
    (x_train, y_train), (x_test, y_test) = load_data(raw_mat_file_path_2D_WPD)

    samples_action = obtained_samples_actions(path=model_path)

    obs1_0 = np.zeros((len(y_test), 1))
    obs1_1 = get_action_history(obs1_0, samples_action, 1)



    _, _, extracted_dimensional_feature, fc1_0, _ = net(torch.tensor((x_test).reshape(-1, 2, 64, 64)).float().to(device),
                                              torch.tensor(obs1_0).float().to(device))

    _, _, extracted_dimensional_feature, fc1_1, MSIF = net(torch.tensor((x_test).reshape(-1, 2, 64, 64)).float().to(device),
                                              torch.tensor(obs1_1).float().to(device))


    savemat(path_tsne + 'tsne_mats/' + 'label.mat', {'label': np.argmax(y_test, axis=1)})

    mode_name = 'MSIL_DARL_'

    plot_with_labels(path_tsne, x_test_1D.reshape(len(x_test_1D), -1), y_test, 2, mode_name+'raw_test_data_2_dimension')
    plot_with_labels(path_tsne, x_test_1D.reshape(len(x_test_1D), -1), y_test, 3, mode_name+'raw_test_data_3_dimension')

    plot_with_labels(path_tsne, x_test.reshape(len(x_test), -1), y_test, 2, mode_name+'WPD_2_dimension')
    plot_with_labels(path_tsne, x_test.reshape(len(x_test), -1), y_test, 3, mode_name+'WPD_3_dimension')

    plot_with_labels(path_tsne, MSIF.data.cpu().numpy(), y_test, 2, mode_name+'MSIF_2_dim')
    plot_with_labels(path_tsne, MSIF.data.cpu().numpy(), y_test, 3, mode_name+'MSIF_3_dim')

    plot_with_labels(path_tsne, extracted_dimensional_feature.data.cpu().numpy(), y_test, 2, mode_name+'extracted_ResNet50_2_dim')
    plot_with_labels(path_tsne, extracted_dimensional_feature.data.cpu().numpy(), y_test, 3, mode_name+'extracted_ResNet50_3_dim')

    plot_with_labels(path_tsne, fc1_0.data.cpu().numpy(), y_test, 2, mode_name+'fc1_0step_2_dim')
    plot_with_labels(path_tsne, fc1_0.data.cpu().numpy(), y_test, 3, mode_name+'fc1_0step_3_dim')

    plot_with_labels(path_tsne, fc1_1.data.cpu().numpy(), y_test, 2, mode_name+'fc1_1step_2_dim')
    plot_with_labels(path_tsne, fc1_1.data.cpu().numpy(), y_test, 3, mode_name+'fc1_1step_3_dim')


    print("=" * 50)


if __name__ == "__main__":
    raw_mat_file_path_1D_signal = '../HDX-800_compound_fault_11class_'+'fanAndDri'+'-noMeanStd' + '.mat'
    raw_mat_file_path_2D_WPD = '../HDX-800_WPD_matrix.mat'
    epochs = 10
    train_type = 0
    current = True
    train_type_list = ['/pre_trained/', '/fine_tuning/']
    for epoch_th in range(epochs):
        print("=========================================" + str(epoch_th + 1) + "=========================================")
        model_path = 'results/'+str(epoch_th+1)+train_type_list[train_type]

        main(model_path, raw_mat_file_path_1D_signal, raw_mat_file_path_2D_WPD, epoch_th+1, current)
        print("=========================================" + "end" + "=========================================")

