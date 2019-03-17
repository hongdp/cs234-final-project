from __future__ import print_function, division

import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
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
    regrets = [0] * dataset.size()
    avg_precision = [0] * dataset.size()

    for i in range(args.shuffle_times):
        agent = get_agent(args.agent_name, config, dataset)
        dataset.shuffle()
        regret = 0
        for ts, data in tqdm(enumerate(dataset)):
            features = data['features']
            label = data['label']
            action, context = agent.act(features)
            reward = calculate_reward(label, action)
            regret -= reward
            agent.feedback(reward, context)
            regrets[ts] += regret
            avg_precision[ts] += 1 - regret/(ts+1)
        print('{} final regret: {} final average precision: {}'.format(i, regret, 1 - regret/dataset.size()))

    regrets = [x/args.shuffle_times for x in regrets]
    avg_precision = [x/args.shuffle_times for x in avg_precision]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(dataset.size()), regrets, 'b')
    fig.savefig("data/scores/{}-regret.png".format(args.agent_name))

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(range(dataset.size()), avg_precision, 'b')
    fig2.savefig("data/scores/{}-avg.png".format(args.agent_name))




if __name__ == '__main__':
    main()



