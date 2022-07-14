from mnist_env import MNISTEnv
from actor_critic_agent import ActorCriticNNAgent
import torch, sys, time
import numpy as np
from itertools import count

# ?????????????????????????, ????????

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass




def main(path):
    # calculate test average reward
    print("=" * 50)
    print("Testing...")

    # initialize agent

    num = 100
    path_actor = path+'net_parameters/' + str(num) + '_actor.pkl'
    path_critic = path+'net_parameters/' + str(num) + '_critic.pkl'
    actor_net = torch.load(path_actor, map_location='cpu')
    critic_net = torch.load(path_critic, map_location='cpu')

    agent = ActorCriticNNAgent(Acotr_Net=actor_net, Critic_Net=critic_net)

    env = MNISTEnv(type='test', seed=None)

    print(len(env.X))

    # ????????

    samples_action = []
    reward_all = []

    keys=[]
    for i in range(2):
        for j in range(3):
            keys.append('step'+str(i+1)+'_label'+str(j+1))
    values = [0]*6
    steps_labels = dict(zip(keys,values))
    # print(steps_labels)  # {'step1_label1': 0, 'step1_label2': 0, 'step2_label1': 0, 'step2_label2': 0}
    # print(steps_labels.keys())  # dict_keys(['step1_label1', 'step1_label2', 'step2_label1', 'step2_label2'])
    # print(steps_labels.values())  # dict_values([0, 0, 0, 0])


    for i in range(len(env.X)):
        rewards = []
        # ??????????????
        state = env.reset(i)

        for t in count():
            action, action_prob, loss_reward = agent.select_action(state, env)
            # print(action, action_prob)  # 8 0.35509148240089417
            label_num = 0 if action in [0, 1, 2, 3] else 1 if action == 4 else 2

            steps_labels['step'+str(t+1)+'_label'+str(label_num+1)]+=1

            next_state, reward, done, _ = env.step(action, loss_reward)
            state = next_state
            rewards.append(reward)
            if done:
                break

        reward_all.append(rewards)

        samples_action.append(env.B)


    print("test_accuracy: %.3f" % (np.mean(reward_all)))
    print(steps_labels)
    print(steps_labels.items())

    print("=" * 50)
    return np.mean(reward_all)


if __name__ == "__main__":
    epochs = 10
    train_type = 0
    train_type_list = ['/pre_trained/', '/fine_tuning/']
    accuracy = []
    for epoch_th in range(epochs):
        path = 'results/' + str(epoch_th + 1) + train_type_list[train_type]
        file = open("./results/" + str(epoch_th + 1) + train_type_list[train_type] + str(epoch_th + 1) + "_infer_result.txt", "w")
        file.close()

        with open("./results/" + str(epoch_th + 1) + train_type_list[train_type] + str(epoch_th + 1) + "_infer_result.txt",
                  "w+") as file:
            print("Training...")
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            print("=========================================" + str(
                epoch_th + 1) + "=========================================")
            terminal = sys.stdout
            sys.stdout = file
            print("=========================================" + str(
                epoch_th + 1) + "=========================================")
            accuracy_epochth = main(path)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

            sys.stdout = terminal
            file.seek(0)
            while 1:
                lines = file.readlines(10000000)
                if not lines:
                    break
                for line in lines:
                    line = line.strip('\n')
                    # if line=='\n':
                    #     continue
                    print(line)  # do something
            print("=========================================" + "end" + "=========================================")
            print("=========================================" + "end" + "=========================================")

            accuracy.append(accuracy_epochth)

    sys.stdout = Logger(train_type_list[train_type][1:-1]+'_infer_total_mean_result.txt')
    print('=' * 60)
    print('=' * 60)
    print(accuracy)
    print('minimum value: ', np.min(accuracy))
    print('average value: ', np.mean(accuracy))
    print('maximum value: ', np.max(accuracy))
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))






'''


'''







