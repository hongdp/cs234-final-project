from __future__ import print_function, division

import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from config import get_config
from agents import get_agent, Actions
from dataset import WarfarinDataSet

def calculate_class_distance_reward(label, action):
    if label < 21:
        if action == Actions.LOW:
            return 0
        elif action == Actions.MEDIUM:
            return -1
        elif action == Actions.HIGH:
            return -2
    elif label < 49:
        if action == Actions.LOW:
            return -1
        elif action == Actions.MEDIUM:
            return 0
        elif action == Actions.HIGH:
            return -1
    else:
        if action == Actions.LOW:
            return -2
        elif action == Actions.MEDIUM:
            return -1
        elif action == Actions.HIGH:
            return 0

def calculate_default_reward(label, action):
    if label < 21:
        return 0 if action == Actions.LOW else -1
    elif label < 49:
        return 0 if action == Actions.MEDIUM else -1
    else:
        return 0 if action == Actions.HIGH else -1

def get_reward_func(name='default'):
    if name == 'default':
        return calculate_default_reward
    elif name == 'class_distance':
        return calculate_class_distance_reward


def is_correct_action(label, action):
    if label < 21:
        return True if action == Actions.LOW else False
    elif label < 49:
        return True if action == Actions.MEDIUM else False
    else:
        return True if action == Actions.HIGH else False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_name', '-a', dest='agent_name', action='store', required=True,
                        help='Name of agent to be used to retieve config and agent object.')
    parser.add_argument('--shuffle_times', '-s', dest='shuffle_times', type=int, action='store', required=True,
                        help='Times to shuffle the dataset.')
    parser.add_argument('--reward_func', '-r', dest='reward_func', default='default', action='store',
                        help='Reward function.')
    parser.add_argument('--output_name', '-o', dest='output_name', default='', action='store',
                        help='prefix of output score files.')


    args = parser.parse_args()
    config = get_config(args.agent_name)
    dataset = WarfarinDataSet(config)
    regrets = np.zeros((args.shuffle_times, dataset.size()))
    precision  = np.zeros((args.shuffle_times, dataset.size()))
    reward_func = get_reward_func(args.reward_func)

    for i in range(args.shuffle_times):
        agent = get_agent(args.agent_name, config, dataset)
        dataset.shuffle()
        regret = 0
        corrects = 0
        for ts, data in tqdm(enumerate(dataset)):
            features = data['features']
            label = data['label']
            action, context = agent.act(features)
            reward = reward_func(label, action)
            agent.feedback(reward, context)

            # Calacualte Eval metrics
            regret -= reward
            regrets[i][ts] = regret

            if is_correct_action(label, action):
                corrects += 1
            precision[i][ts] = corrects/(ts+1)
        print('{} final regret: {} final average precision: {}'.format(i, regret, precision[i][-1]))

    if args.output_name:
        output_name = args.output_name
    else:
        output_name = args.agent_name
    avg_regrets = np.average(regrets, axis=0)
    avg_precision = np.average(precision, axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(dataset.size()), avg_regrets, 'b')
    fig.savefig("data/scores/{}-regret.png".format(output_name))
    print(np.std(regrets, axis=0))
    with open("data/scores/{}-regret-values.txt".format(output_name), mode='w') as f:
        f.write(';'.join(map(lambda x: ','.join(map(str, x)), regrets)))

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(range(dataset.size()), avg_precision, 'b')
    fig2.savefig("data/scores/{}-precision.png".format(output_name))
    with open("data/scores/{}-precision-values.txt".format(output_name), mode='w') as f:
        f.write(';'.join(map(lambda x: ','.join(map(str, x)), precision)))






if __name__ == '__main__':
    main()



