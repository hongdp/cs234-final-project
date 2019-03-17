from __future__ import print_function, division

import argparse
import matplotlib.pyplot as plt
from config import get_config
from agents import get_agent, Actions
from dataset import WarfarinDataSet

def calculate_reward(label, action):
    if label < 21:
        return 0 if action == Actions.LOW else -1
    elif label < 49:
        return 0 if action == Actions.MEDIUM else -1
    else:
        return 0 if action == Actions.HIGH else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_name', '-a', dest='agent_name', action='store', required=True,
                        help='Name of agent to be used to retieve config and agent object.')
    parser.add_argument('--shuffle_times', '-s', dest='shuffle_times', type=int, action='store', required=True,
                        help='Times to shuffle the dataset.')


    args = parser.parse_args()
    config = get_config(args.agent_name)
    dataset = WarfarinDataSet(config)
    agent = get_agent(args.agent_name, config, dataset.vocab)
    regrets = [0] * dataset.size()
    avg_regrets = [0] * dataset.size()

    for i in range(args.shuffle_times):
        dataset.shuffle()
        regret = 0
        for ts, data in enumerate(dataset):
            features = data['features']
            label = data['label']
            action = agent.act(features)
            reward = calculate_reward(label, action)
            regret -= reward
            agent.feedback(features, reward)
            regrets[ts] += regret
            avg_regrets[ts] += regret/(ts+1)
        precision = 1 - (regret / dataset.size())
        print('{} final regret: {} final average precision: {}'.format(i, regret, precision))

    regrets = [x/args.shuffle_times for x in regrets]
    avg_regrets = [x/args.shuffle_times for x in avg_regrets]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(dataset.size()), regrets, 'b')
    fig.savefig("data/scores/{}-regret.png".format(args.agent_name))

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(range(dataset.size()), avg_regrets, 'b')
    fig2.savefig("data/scores/{}-avg.png".format(args.agent_name))




if __name__ == '__main__':
    main()



