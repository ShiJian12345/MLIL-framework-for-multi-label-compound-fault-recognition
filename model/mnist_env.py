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
    print(f.keys()) 

    x_train, y_train = f['x_WPD_train'], f['train_label']

    x_test, y_test = f['x_WPD_test'], f['test_label']



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

        self.h = 64
        self.w = 64

        self.action_space = spaces.Discrete(7)

        self.A = []
        self.B = []
        self.C = []
        self.D = []

        self.E = []



    def step(self, action, loss_reward):

        self.B.append(action)

        self.C = [i for i in self.A if i not in self.B]
        assert (self.action_space.contains(action))
        reward = -1


        true_label = self.D

        if self.type == 'train':
            self.cross_entropy.append(reward * self.num * (-1))
            if action in true_label:

                reward = 2

        else:
            reward = 0
            if action in true_label:
                reward = 1

        self.steps += 1

        obs = self._get_obs()

        done = self.steps >= MAX_STEPS

        info = {}

        return obs, reward, done, info

    def convert_to_one_hot(self, y, num_class):
        array = np.zeros(num_class)
        for i in y:
            array[i]=1
        return array

    def find_true_action(self, array):

        index=[]

        y = np.argmax(array)

        y = label_dic[y]
        self.E = y

        for i, value in enumerate(y):
            if value==1:
                index.append(i)
        return index


    def reset(self, num):

        self.i = num * 1 % self.count


        self.A = [0, 1, 2, 3, 4, 5]

        self.D = self.find_true_action(self.Y[self.i])

        self.B = []

        self.C = [i for i in self.A if i not in self.B]

        self.steps = 0



        return self._get_obs()

    def _get_obs(self):

        obs_0 = self.X[self.i].astype(np.float32)
 
        #if len(self.B)==0:

        #    obs_1 = np.zeros((1,1))

        #elif len(self.B)==1:

        #    obs_1 = np.array([self.B[0]]).reshape(1, -1)

        #elif len(self.B)>=2:

        #    obs_1 = np.array([self.B[-1]]).reshape(1, -1)
        
        if len(self.B)==0:
            obs_1 = np.zeros((1,6))

        elif len(self.B)>=1:
            obs_1 = self.convert_to_one_hot(self.B, 6).reshape(1, -1)

        obs_0 = obs_0.reshape(1, 2, 64, 64)

        return [obs_0, obs_1]

    def shu(self):
        shuffle_index = np.arange(self.count)
        np.random.shuffle(shuffle_index)
        self.X = self.X[shuffle_index]
        self.Y = self.Y[shuffle_index]