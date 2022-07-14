#!/usr/bin/env python
import random
import time
import copy, os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torchvision import models
from TwoDimension_CNN import Discriminator
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

        self.lin0 = nn.Linear(512 + 1, 64)

        self.out_digit = nn.Linear(64, 6)

    def forward(self, x, y):
        x = x.view(-1, 2, self.input_dim, self.input_dim)  # (batch_size, 2, 64, 64)
        # print(x.size())
        y = y.view(-1, 1)
        # x4 = self.resnet(x)
        x4 = self.cnn(x)

        x4 = torch.cat((x4, y), 1)

        x = self.lin0(x4)
        x = F.relu(x)

        x = self.out_digit(x)
        action_prob = F.softmax(x, dim=1)

        return action_prob, x


class Critic(nn.Module):
    def __init__(self, input_dim=64, output_dim=9):
        self.output_dim = output_dim
        super(Critic, self).__init__()
        self.input_dim = input_dim
        # resnet50 = models.resnet18(pretrained=False)
        # resnet50.fc = nn.Sequential()
        # self.resnet = resnet50

        cnn = Discriminator(output_dim=6)
        cnn.fc = nn.Sequential()
        self.cnn = cnn
        self.lin0 = nn.Linear(512 + 1, 64)

        self.out_critic = nn.Linear(64, 1)

    def forward(self, x, y):
        x = x.view(-1, 2, self.input_dim, self.input_dim)  # (batch_size, 2, 64, 64)
        # print(x.size())
        y = y.view(-1, 1)
        x4 = self.cnn(x)

        x4 = torch.cat((x4, y), 1)

        x = self.lin0(x4)
        x = F.relu(x)

        value = self.out_critic(x)

        return value


def torch_to_numpy(tensor):
    if torch.cuda.is_available():
        return tensor.data.cpu().numpy()
    return tensor.detach().numpy()



def numpy_to_torch(array):

    return torch.tensor(array).float().to(device)


gamma = 0.1
class ActorCriticNNAgent():
    # gamma = 0.99
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 5
    buffer_capacity = 1000
    batch_size = 2*11
    actor_lr = 6e-6
    critic_lr = 2e-5
    # batch_size = 2*4

    def __init__(self, Acotr_Net=None, Critic_Net=None, Train=False, Pre_Train=False, file_folder_path=None):
        super(ActorCriticNNAgent, self).__init__()
        if Acotr_Net==None and Critic_Net==None:
            self.actor_net = Actor()
            self.critic_net = Critic()
            if Train and Pre_Train:
                path_actor = file_folder_path + '_actor.pkl'
                path_critic = file_folder_path + '_critic.pkl'
                actor_net = torch.load(path_actor, map_location='cpu')
                critic_net = torch.load(path_critic, map_location='cpu')
                self.actor_net = actor_net
                self.critic_net = critic_net
            elif Train and not Pre_Train:
                # for k, v in self.actor_net.named_parameters():
                #     print(k)
                # print('='*50)
                # net = torch.load('../pre_trained_ResNet50/figures_multi_model_save/1/The_99_epoch_model_parameters.pkl',map_location='cpu')
                net = torch.load(file_folder_path+'_epoch_model_parameters.pkl',map_location='cpu')
                # for k, v in net.named_parameters():
                #     print(k)
                net.pop('fc.0.weight')
                net.pop('fc.0.bias')
                net.pop('fc.2.weight')
                net.pop('fc.2.bias')
                actor_net_dict = self.actor_net.state_dict()
                critic_net_dict = self.critic_net.state_dict()
                net_dict = {'cnn.'+k: v for k, v in net.items()}
                actor_net_dict.update(net_dict)
                critic_net_dict.update(net_dict)


                self.actor_net.load_state_dict(actor_net_dict)
                self.critic_net.load_state_dict(critic_net_dict)

                self.actor_net.cnn.eval()
                self.critic_net.cnn.eval()
                # for param in self.actor_net.resnet.parameters():
                #     param.requires_grad = False
                # for param in self.critic_net.resnet.parameters():
                #     param.requires_grad = False
                for k, v in self.actor_net.named_parameters():
                    if 'cnn' in k:
                        # print(k)
                        v.requires_grad = False
                for k, v in self.critic_net.named_parameters():
                    if 'cnn' in k:
                        v.requires_grad = False

                # actor_net_dict = self.actor_net.state_dict()
                # print(type(actor_net_dict))
                #
                # # actor_net_dict.pop('lin1.weight')
                # # actor_net_dict.pop('lin1.bias')
                # # actor_net_dict.pop('out_digit.weight')
                # # actor_net_dict.pop('out_digit.bias')
                # # actor_net_dict.pop('out_critic.weight')
                # # actor_net_dict.pop('out_critic.bias')
                # critic_net_dict = self.critic_net.state_dict()
                # # for i in ['lin1.weight', 'lin1.bias', 'out_digit.weight', 'out_digit.bias', 'out_critic.weight', 'out_critic.bias']:
                # #     actor_net_dict.pop(i)
                # #     critic_net_dict.pop(i)
                # miss_list = ['lin1.weight', 'lin1.bias', 'out_digit.weight', 'out_digit.bias', 'out_critic.weight',
                #  'out_critic.bias']
                # # for k, v in net.items():
                # #     print(k)
                # net_actor = {k: v for k, v in net.items() if k in actor_net_dict and k not in miss_list}
                # net_critic = {k: v for k, v in net.items() if k in critic_net_dict and k not in miss_list}
                # actor_net_dict.update(net_actor)
                # critic_net_dict.update(net_critic)
                # self.actor_net.load_state_dict(actor_net_dict)
                # self.critic_net.load_state_dict(critic_net_dict)
                # # print(self.actor_net)
                # # for param in self.actor_net.parameters():
                # #     param.requires_grad = False
                #
                # for k, v in self.actor_net.named_parameters():
                #     # print(k)
                #     if k not in miss_list:
                #         v.requires_grad = False
                # for k, v in self.critic_net.named_parameters():
                #     # print(k)
                #     if k not in miss_list:
                #         v.requires_grad = False
                                # self.actor_net.load_state_dict(net)
                # self.critic_net.load_state_dict(net)
                #
                # pretrained_dict = torch.load(model_weight)
                # model_dict = myNet.state_dict()
                # # 1. filter out unnecessary keys
                # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                # # 2. overwrite entries in the existing state dict
                # model_dict.update(pretrained_dict)
                # myNet.load_state_dict(model_dict)


        else:
            self.actor_net = Acotr_Net
            self.critic_net = Critic_Net
        # self.actor_net = self.actor_net.to(device)
        # self.critic_net = self.critic_net.to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        # self.writer = SummaryWriter('../exp')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), self.actor_lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), self.critic_lr)


    def select_action(self, state, env):
        # state = torch.from_numpy(state).float().unsqueeze(0)
        # state = numpy_to_torch(state)
        # print(state[0].size(),state[1].size())  # torch.Size([1, 3, 64, 64]) torch.Size([1, 6])
        with torch.no_grad():
            # action_prob_, actions_prob = self.actor_net(numpy_to_torch(state[0]), numpy_to_torch(state[1]))
            action_prob_, actions_prob = self.actor_net(torch.tensor(state[0]).float(), torch.tensor(state[1]).float())

        # print(action_prob_)

        # zui da zhi
        action_prob = torch_to_numpy(action_prob_).flatten()
        # 从未选择动作集合C中整理出概率
        _action_prob = [action_prob[i] for i in env.C]
        # 从未选择动作集合C中找出动作概率最大的位置
        a_temp = np.argmax(np.array(_action_prob))
        # 以得到的概率最大位置得到概率值，再从整个集合中找出动作指引
        action = list(action_prob).index(_action_prob[a_temp])

        # sui ji cai yang
        # c = Categorical(numpy_to_torch(_action_prob))
        # a_temp = c.sample()
        # # print(action_prob, _action_prob[a_temp.item()])
        # action = list(action_prob).index(_action_prob[a_temp.item()])



        Sigmoid_fun = nn.Sigmoid()
        criterion = nn.BCELoss()

        # print(y1.size(), torch.FloatTensor(env.E).size())

        loss = criterion(Sigmoid_fun(actions_prob), torch.FloatTensor(env.E).view(1,-1))
        # print(loss)
        return action, action_prob_[:, action].item(), torch.tensor(loss).float()

    def get_value(self, state):
        # state = torch.from_numpy(state)

        state = numpy_to_torch(state)

        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), './param/net_param/actor_net' + str(time.time())[:10], +'.pkl')
        torch.save(self.critic_net.state_dict(), './param/net_param/critic_net' + str(time.time())[:10], +'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        # state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        # for t in self.buffer:
        #     print(t.state[0].size())
        state0 = torch.tensor([t.state[0] for t in self.buffer], dtype=torch.float)
        state1 = torch.tensor([t.state[1] for t in self.buffer], dtype=torch.float)

        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).to(device)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float).to(device)
        # print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print('I_ep {} train {} times'.format(i_ep, self.training_step))
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state0[index].to(device), state1[index].to(device))
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                # print(type(state[index]))
                # print(state[index].shape)
                action_prob = self.actor_net(state0[index].to(device), state1[index].to(device))[0].gather(1, action[index].to(device))  # new policy

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:]  # clear experience

    def copy(self):

        # create a copy of this agent with frozen weights
        agent = ActorCriticNNAgent()
        agent.actor_net = copy.deepcopy(self.actor_net)
        agent.critic_net = copy.deepcopy(self.critic_net)
        agent.actor_net.cnn.eval()
        agent.critic_net.cnn.eval()
        # agent.actor_net.eval()
        # agent.critic_net.eval()
        # agent.trainable = False
        for param in agent.actor_net.parameters():
            param.requires_grad = False
        for param in agent.critic_net.parameters():
            param.requires_grad = False

        return agent