import numpy as np

class ClinicalValueFunc():

    def __init__(self, vocab):
        '''
        Initialize Value function of Clinical dosing algorithm.
        Use vocabulary to dyanmically generate weights to avoid mismatching when
        vocabulary and feature being regenerated.

        Expected input features are: [Race: Enum(5), Age: Enum(11), Height: Float(1),
        Weight: Float(1), Amiodarone: Enum(4), Carbamazepine: Enum(4), Phenytoin: Enum(4), Rifampin or Rifampicin: Enum(4)]

        Params:
            vocab: enum vocabulary of form:
                {feature: {val0: 0, val1: 1, ...}, ...}
        '''
        self.bias = 4.0376
        self.weights = np.zeros([34], dtype=np.float32)


    def eval(self, feature):
        return (np.matmul(feature, self.weights) + self.bias) ** 2

