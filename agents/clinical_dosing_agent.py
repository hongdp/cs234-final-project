import numpy as np
from .agent import Agent, Actions

class ClinicalDosingAgent(Agent):

    def __init__(self, config, vocab):
        '''
        Initialize Value function of Clinical dosing algorithm.
        Use vocabulary to dyanmically generate weights to avoid mismatching when
        vocabulary and feature being regenerated.

        Expected input features are: [Race: Enum(5), Age: Enum(10), Height: Float(1),
        Weight: Float(1), Amiodarone: Enum(3), Carbamazepine: Enum(3), Phenytoin: Enum(3), Rifampin or Rifampicin: Enum(3)]

        Params:
            vocab: enum vocabulary of form:
                {feature: {val0: 0, val1: 1, ...}, ...}
        '''
        self.vocab = vocab
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

        height_weights = np.asarray([0.0118], dtype=np.float32)
        weight_weights = np.asarray([0.0134], dtype=np.float32)

        amiodarone_weights = np.zeros([3], dtype=np.float32)
        amiodarone_weights[vocab['Amiodarone (Cordarone)']['1']] = -0.5695

        enzyme_weights = np.asarray([1.2799], dtype=np.float32)

        self.weights = np.concatenate((race_weights, age_weights, height_weights, weight_weights, amiodarone_weights, enzyme_weights ))



    def act(self, feature):
        enzyme_mask = np.zeros_like(feature)

        if feature[20+self.vocab['Carbamazepine (Tegretol)']['1']] or feature[23+self.vocab['Phenytoin (Dilantin)']['1']] or feature[26+self.vocab['Rifampin or Rifampicin']['1']]:
            enzyme_feature = np.asarray([1.0], dtype=np.float32)
        else:
            enzyme_feature = np.asarray([0.0], dtype=np.float32)

        feature = np.concatenate((feature[:20], enzyme_feature))
        dose = (np.matmul(feature, self.weights) + self.bias) ** 2
        if dose < 21:
            return Actions.LOW
        elif dose < 49:
            return Actions.MEDIUM
        else:
            return Actions.HIGH

    def feedback(self, feature, reward, action):
        pass
