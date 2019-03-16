import numpy as np
from .agent import Agent, Actions

class ClinicalDosingAgent(Agent):

    def __init__(self, config, vocab):
        '''
        Initialize Value function of Clinical dosing algorithm.
        Use vocabulary to dyanmically generate weights to avoid mismatching when
        vocabulary and feature being regenerated.

        Expected input features are: [Race: Enum(4), Age: Enum(10), Height: Float(1),
        Weight: Float(1), Amiodarone: Enum(3), Carbamazepine: Enum(3), Phenytoin: Enum(3), Rifampin or Rifampicin: Enum(3)]

        Params:
            vocab: enum vocabulary of form:
                {feature: {val0: 0, val1: 1, ...}, ...}
        '''
        self.bias = 4.0376
        race_weights = np.zeros([5], dtype=np.float32)
        race_weights[vocab['Race']['Asian']] = -0.6752
        race_weights[vocab['Race']['Black or African American']] = 0.4060
        race_weights[vocab['Race']['Unknown']] = 0.0443

        age_weights = np.zeros([10], dtype=np.float32)
        age_weights[vocab['Age']['10 - 19']] = -0.2546 * 1
        age_weights[vocab['Age']['20 - 29']] = -0.2546 * 2
        age_weights[vocab['Age']['30 - 39']] = -0.2546 * 3
        age_weights[vocab['Age']['40 - 49']] = -0.2546 * 4
        age_weights[vocab['Age']['50 - 59']] = -0.2546 * 5
        age_weights[vocab['Age']['60 - 69']] = -0.2546 * 6
        age_weights[vocab['Age']['70 - 79']] = -0.2546 * 7
        age_weights[vocab['Age']['80 - 89']] = -0.2546 * 8
        age_weights[vocab['Age']['90+']] = -0.2546 * 9
        age_weights[vocab['Age']['NA']] = -0.2546 * 5

        height_weights = np.asarray([0.0118], dtype=np.float32)
        weight_weights = np.asarray([0.0134], dtype=np.float32)

        amiodarone_weights = np.zeros([3], dtype=np.float32)
        amiodarone_weights[vocab['Amiodarone (Cordarone)']['1']] = -0.5695

        carbamazepine_weights = np.zeros([3], dtype=np.float32)
        carbamazepine_weights[vocab['Carbamazepine (Tegretol)']['1']] = 1.2799

        phenytoin_weights = np.zeros([3], dtype=np.float32)
        phenytoin_weights[vocab['Phenytoin (Dilantin)']['1']] = 1.2799

        rifampin_weights = np.zeros([3], dtype=np.float32)
        rifampin_weights[vocab['Rifampin or Rifampicin']['1']] = 1.2799

        self.weights = np.concatenate((race_weights, age_weights, height_weights, weight_weights, amiodarone_weights, carbamazepine_weights, phenytoin_weights, rifampin_weights))



    def act(self, feature):
        dose = (np.matmul(feature, self.weights) + self.bias) ** 2
        if dose < 21:
            return Actions.LOW
        elif dose < 49:
            return Actions.MEDIUM
        else:
            return Actions.HIGH

    def feedback(self, feature, reward):
        pass
