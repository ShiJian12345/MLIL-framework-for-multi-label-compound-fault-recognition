from mnist_env import MNISTEnv
from actor_critic_agent import ActorCriticNNAgent
import numpy as np
import time, gc
import matplotlib.pyplot as plt
from scipy.io import savemat
import torch, os
from collections import namedtuple
from itertools import count
import sys


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    epochs = 20
    train_type = 0
    epochs_train = 100
    iterations_train = 120
    episodes_train = 11
    seed = 1
    torch.manual_seed(seed)
    accuracy = []
    file_folder_path = None

    train_type_list = ['/pre_trained/', '/fine_tuning/']
    for epoch_th in range(epochs):
        if train_type == 1:
            file_folder_path = './results/' + str(epoch_th + 1) + '/pre_trained/net_parameters/' + str(100)
        elif train_type == 0:
            file_folder_path = '../pre_trained_cnn/pre_trained_2D_cnn/fanAndDri/figures_MLCNN_save/5/The_' + str(99)
        folder_path = './results/'+str(epoch_th+1)
        if not os.path.exists(folder_path):
            for train_type_name in train_type_list:
                os.makedirs(folder_path + train_type_name + "/mat")
                os.makedirs(folder_path + train_type_name + "/net_parameters")
                os.makedirs(folder_path + train_type_name + "/picture")
                file = open("./results/" + str(epoch_th + 1) + train_type_name + str(epoch_th + 1) + "_result.txt", "w")
                file.close()
        plt.close()
        with open("./results/" + str(epoch_th + 1) + train_type_list[train_type] + str(epoch_th + 1) + "_result.txt", "w+") as file:
            print("Training...")
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            print("=========================================" + str(epoch_th + 1) + "=========================================")
            terminal = sys.stdout
            print("=========================================" + str(epoch_th + 1) + "=========================================")
            trained_agent = train(epoch_th + 1, epochs_train, iterations_train, episodes_train, train_type, file_folder_path)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

            sys.stdout = terminal
            file.seek(0)
            while 1:
                lines = file.readlines(10000000)
                if not lines:
                    break
                for line in lines:
                    line = line.strip('\n')
                    print(line)  
            print("=========================================" + "end" + "=========================================")
            print("=========================================" + "end" + "=========================================")

            accuracy.append(trained_agent)

    sys.stdout = Logger("total_mean_result.txt")
    print('='*60)
    print('='*60)
    print(accuracy)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))



def train(result_num, epochs, iterations, episodes, Pre_Train, file_folder_path):
    reletive_path = './results/' + str(result_num)
    train_type = '/pre_trained/'
    if Pre_Train:
        train_type = '/fine_tuning/'
    agent = ActorCriticNNAgent(Train=True, Pre_Train=Pre_Train, file_folder_path=file_folder_path)
    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

    env = MNISTEnv(type='train', seed=None)
    start = time.time()
    accuracy_epochs = []
    reward_loss_epochs = []
    cross_entropy_epochs = []
    for epoch in range(epochs):
        print('epoch:%d' % (epoch))
        if epoch<0:
            torch.save(agent.actor_net, reletive_path + train_type + 'net_parameters/' + str(epoch) + '_actor.pkl')
            torch.save(agent.actor_net.state_dict(), reletive_path + train_type + 'net_parameters/' + str(epoch) + '_actor_parameters.pkl')
            torch.save(agent.critic_net, reletive_path + train_type + 'net_parameters/' + str(epoch) + '_critic.pkl')
            torch.save(agent.critic_net.state_dict(), reletive_path + train_type + 'net_parameters/' + str(epoch) + '_critic_parameters.pkl')
        reward_loss_single_epoch = []
        cross_entropy_single_epoch = []
        for iter in range(iterations):
            if iter % 10 == 0: print("Starting iteration %d" % (iter))
            rewards = []
            cross_entropys = []
            for ep in range(episodes):
                state = env.reset(iter * episodes + ep)
                # print(state)
                total_reward = 0
                for t in count():
                    action, action_prob, loss_reward = agent.select_action(state, env)
                    cross_entropys.append(loss_reward)
                    # print(action, action_prob)
                    next_state, reward, done, _ = env.step(action, loss_reward)
                    trans = Transition(state, action, action_prob, reward, next_state)

                    agent.store_transition(trans)
                    state = next_state
                    total_reward += reward
                    if done:
                        break
                rewards.append(total_reward)
            reward_loss_single_epoch.append(rewards)
            cross_entropy_single_epoch.append(np.mean(cross_entropys))
            agent.update(iter)
            if iter % 10 == 0:
                print("Mean total reward / episode: %.3f" % np.mean(rewards))
        print('='*50)
        print("In final epoch, mean total reward / epoch: %.3f" % np.mean(reward_loss_single_epoch))
        print('='*50)
        reward_loss_epochs.append(np.mean(reward_loss_single_epoch))
        cross_entropy_epochs.append(np.mean(cross_entropy_single_epoch))
        if (epoch+1) % 1 == 0:
            torch.save(agent.actor_net, reletive_path + train_type + 'net_parameters/' + str(epoch+1) + '_actor.pkl')
            torch.save(agent.actor_net.state_dict(), reletive_path + train_type + 'net_parameters/' + str(epoch+1) + '_actor_parameters.pkl')
            torch.save(agent.critic_net, reletive_path + train_type + 'net_parameters/' + str(epoch+1) + '_critic.pkl')
            torch.save(agent.critic_net.state_dict(), reletive_path + train_type + 'net_parameters/' + str(epoch+1) + '_critic_parameters.pkl')
        env.shu()
    end = time.time()

    print("Completed %d epoches %d iterations of %d episodes in %.3f s" % (epochs, iterations, episodes, end - start))


    final_test_acc = test(reletive_path, agent.copy())

    return final_test_acc


def test(reletive_path, agent):

    print("=" * 50)
    print("Testing...")


    agent = agent
    env = MNISTEnv(type='test', seed=None)


    samples_action = []
    reward_all = []
    for i in range(len(env.X)):
        rewards = []
        state = env.reset(i)
        done = False
        while not done:
            action, action_prob, loss_reward = agent.select_action(state, env)
            # print(action, action_prob)
            next_state, reward, done, _ = env.step(action, loss_reward)
            state = next_state
            rewards.append(reward)

        reward_all.append(rewards)

        samples_action.append(env.B)

    acc = np.mean(reward_all)
    print("test_accuracy: %.3f" % (acc))
    print("=" * 50)
    return acc


if __name__ == '__main__':
    main()
    print("end")