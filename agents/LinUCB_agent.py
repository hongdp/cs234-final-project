import numpy as np
import random
from .agent import Agent, Actions


class LinUCBAgent(Agent):
    def __init__(self, config, dataset):
        self.tdim = len(dataset.examples)
        print(self.tdim)
        self.dim = len(dataset.examples[0]['features'])
        print(self.dim)
        # self.history = []
        # self.total = 0
        self.A = np.stack([np.identity(self.dim) for _ in range(3)], axis=0)
        self.b = np.zeros((3,self.dim))
        sigma = 0.4
        self.alpha = 1+ np.sqrt(np.log(2*3*self.tdim/sigma)/2)
        print(self.alpha)
        #pass

    def act(self, feature):
        p = np.zeros(3)
        for ind in range(3):
            invA = np.linalg.inv(self.A[ind])
            theta = np.matmul(invA, self.b[ind])
            # norm = np.linalg.norm(theta)
            # if norm != 0:
            #     theta = theta/norm
            # print(np.matmul(feature, theta), self.alpha * np.sqrt(np.matmul(np.matmul(invA, feature), feature)))
            p[ind] = np.matmul(feature, theta) + self.alpha * np.sqrt(np.matmul(np.matmul(invA, feature), feature))

        action = np.argmax(p)
        # actions = np.argwhere(p == np.amax(p)).flatten()
        # print(p, actions)
        # action = actions[0]
        # action = actions[random.randint(0,len(actions)-1)]

        if action == 0:
            return Actions.LOW, {'feature': feature, 'action': action}
        elif action == 1:
            return Actions.MEDIUM, {'feature': feature, 'action': action}
        else:
            return Actions.HIGH, {'feature': feature, 'action': action}
        #pass, 'action': action

    def feedback(self, reward, context):
        # print(reward, action.value)
        feature = context['feature']
        ind = context['action']
        # print(np.outer(feature, feature)
        self.A[ind] += np.outer(feature, feature)
        self.b[ind] += reward * feature
        #pass