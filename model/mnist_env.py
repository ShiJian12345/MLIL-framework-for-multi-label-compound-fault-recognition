import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

MAX_STEPS = 2

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

fault_list_label = ['O', 'I', 'B', 'R', 'OI', 'OB', 'OR', 'IB', 'IR', 'BR', 'N']


mat_file = '../HDX-800_WPD_matrix.mat'
def load_data(path=mat_file):
    f = loadmat(path)
    print(f.keys())  # dict_keys(['__header__', '__version__', '__globals__', 'train_data', 'train_label', 'test_data', 'test_label'])

    x_train, y_train = f['x_WPD_train'], f['train_label']

    x_test, y_test = f['x_WPD_test'], f['test_label']
    # print(type(x_train))  # <class 'numpy.ndarray'>

    # print('x_train', x_train.shape)  # x_train (1320, 2, 64, 64)
    # print('y_train', y_train.shape)   # y_train (1320, 11)
    # print('x_test', x_test.shape)  # x_test (330, 2, 64, 64)
    # print('y_test', y_test.shape)  # y_test (330, 11)


    return (x_train, y_train), (x_test, y_test)


class MNISTEnv(gym.Env):

    def __init__(self, type='train', seed=2069):

        if seed:
            np.random.seed(seed=seed)

        self.type = type
        (x_train, y_train), (x_test, y_test) = load_data()
        self.num=1
        if self.type == 'train':
            self.X = x_train
            self.Y = y_train
            self.count = len(y_train)
            self.cross_entropy = []

        elif self.type == 'test':
            self.X = x_test
            self.Y = y_test
            self.count = len(y_test)
            self.pred_label_memory=[]
            self.true_label_memory=[]
            self.fault_list=fault_list_label

        # print(self.X.shape) # train(2160, 4096) test(540, 4096)
        # print(self.Y.shape) # test(2160, 9) test(540, 9)
        self.h = 64
        self.w = 64

        # action is an integer in {0, ..., 39}
        # see 'step' for interpretation
        self.action_space = spaces.Discrete(7)
        # self.observation_space = spaces.Box(0, 255, [self.h, self.w])
        self.A = []
        self.B = []
        self.C = []
        self.D = []

        self.E = []

        # epoch = 17
        # net = torch.load('/opt/TP/coder/paper/duanjie_4speed_3type_2load/pre_trained_ResNet50/figures_multi_model_save/1/The_' + str(epoch) + '_epoch_model.pkl', map_location='cpu')
        # net.fc = nn.Sequential()
        # for param in net.parameters():
        #     param.requires_grad = False
        # self.model = net


    def step(self, action, loss_reward):
        # print(action)
        # 记录历史动作，每一个动作都放入历史动作集合中
        self.B.append(action)
        # print(action, self.i)
        self.C = [i for i in self.A if i not in self.B]
        # if self.type == 'train':
        #     self.C = [i for i in self.A if i not in self.B]
        # elif self.type == 'test':
        #     if action in [0, 1, 2]:
        #         self.C = [i for i in self.C if i not in [0, 1, 2]]
        #     elif action in [3, 4]:
        #         self.C = [i for i in self.C if i not in [3, 4]]

        # action a consists of
        #   1. direction in {N, S, E, W}, determined by = a (mod 4)
        #   2. predicted class (0-9), determined by floor(a / 4)
        assert (self.action_space.contains(action))
        # reward = -0.5-loss_reward
        reward = -1
        # print(loss_reward)
        # reward = -loss_reward

        # reward = -np.mean(np.square(np.subtract(pred, self.Y[self.i + self.steps])))
        # reward = -np.mean(-np.sum(self.Y[self.i] * np.log(pred+10**(-5))))/self.num
        # Sigmoid_fun = nn.Sigmoid()
        # loss = criterion(Sigmoid_fun(outputs), labels)


        true_label = self.D
        # print(action, true_label)
        # print(self.i + self.steps)
        if self.type == 'train':
            self.cross_entropy.append(reward * self.num * (-1))
            if action in true_label:

                # if self.num >= 20:
                #     reward += 1
                # else:
                reward = 2
            # label = self.convert_to_one_hot(self.Y[self.i + self.steps], 10)
            # reward = -np.sum(np.square(np.subtract(pred, label)))

        else:
            reward = 0
            if action in true_label:
                reward = 1
            # self.pred_label_memory.append(self.fault_list[action])
            # self.true_label_memory.append(self.fault_list[true_label])

        self.steps += 1

        # state (observation) consists of masked image (h x w)
        obs = self._get_obs()

        done = self.steps >= MAX_STEPS

        # info is empty (for now)
        info = {}

        return obs, reward, done, info

    def convert_to_one_hot(self, y, C):
        return np.eye(C)[y]

    def find_true_action(self, array):
        '''
        here is to find the location at which the value is equal to 1. [0 1 0 1 0 0] -> [1, 3]
        '''
        index=[]

        #  1. first find the single label
        y = np.argmax(array)
        # print(y)

        # 2. according to the single label, then find the multi-label value
        y = label_dic[y]
        self.E = y
        # 3. according to the multi-label label, find the location at which the value is equal to 1

        for i, value in enumerate(y):
            if value==1:
                index.append(i)
        return index


    def reset(self, num):
        # resets the environment and returns initial observation
        # zero the mask, move to random location, and choose new image

        # initialize at random location or image center
        # self._reveal()
        # self.i = np.random.randint(self.n)

        # 更新当前epoch的指引，如果是第二个epoch的指引，则求对数据集长度的余数
        self.i = num * 1 % self.count


        # 每一次探索一个环境，都要重新重置下 所有动作集合 A, 已选动作集合 B, 未选动作集合 C, 该环境的正确动作 D
        # 更新该样本所有可能的动作
        self.A = [0, 1, 2, 3, 4, 5]

        # 上面更新每次环境的样本，这里更新每次该样本的正确动作集 D
        self.D = self.find_true_action(self.Y[self.i])

        # 集合B是已经选择过的动作
        self.B = []

        # 集合C是没有选择过的动作
        self.C = [i for i in self.A if i not in self.B]

        # 重置每一回合中的第一步
        self.steps = 0



        return self._get_obs()

    def _get_obs(self):
        # if self.i + self.steps == self.count:
        #     return -1

        # obs = self.X[self.i + self.steps].reshape(1, -1)
        # assert self.observation_space.contains(obs)
        # print(self.B)
        obs_0 = self.X[self.i].astype(np.float32)
        # print('obs_0:', obs_0.shape)  # (64, 64, 3)
        if len(self.B)==0:
            # obs_1 = np.zeros((1,18))
            obs_1 = np.zeros((1,1))

        elif len(self.B)==1:
            # obs_1 = np.concatenate( ( self.convert_to_one_hot(self.B[-1], 9).reshape(1, -1),
            #                           np.zeros((1,9))),
            #                         axis=1)

            obs_1 = np.array([self.B[0]]).reshape(1, -1)
            # obs_1 = self.convert_to_one_hot(self.B[-1], 9).reshape(1, -1)

        elif len(self.B)>=2:
            # obs_1 = np.concatenate( ( self.convert_to_one_hot(self.B[-2], 9).reshape(1, -1),
            #                           self.convert_to_one_hot(self.B[-1], 9).reshape(1, -1) ),
            #                         axis=1)

            obs_1 = np.array([self.B[-1]]).reshape(1, -1)
        # obs_0 = self.model(torch.tensor(obs_0.reshape(1, 1, 1, 4096)))
        obs_0 = obs_0.reshape(1, 2, 64, 64)

        # print('obs_0_model', obs_0.shape)  # torch.Size([1, 2048])

        # obs = np.concatenate((obs_0.detach().numpy().reshape(1, -1), obs_1), axis=1)

        return [obs_0, obs_1]
