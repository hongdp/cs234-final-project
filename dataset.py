import random
import numpy as np
import config

from enum import Enum
from csv import reader

class FeatureTypes(Enum):
    UNKNOWN = 0 # Feature will be ignored.
    ENUM = 1 # Feature value will be treated as enum values and turned into one hot representation.
    FLOAT = 2 # Feature value will be directly treated as float values.
    LABEL = 3 # Feature value will be used as label. dtype of label is float32.

def BuildEnumVocab(config, output=False):
    '''
    Build vocabulary for enum features requested in DataSetConfig.
    The vocabulary will be dumped to the vocab file specified in config if output is ture.

    vocab file format: feature, val0, val1, val2\n
    '''

    with open(config.data_filename , mode='r', newline='') as data_file:
        # Build feature types list.
        csv_reader = reader(data_file)
        cols = next(csv_reader)
        vocab = {}
        feature_types = []
        for col in cols:
            if col in config.enum_feature_cols:
                feature_types.append(FeatureTypes.ENUM)
                vocab[col] = set()
            elif col in config.float_feature_cols:
                feature_types.append(FeatureTypes.FLOAT)
            elif col == config.label_col:
                feature_types.append(FeatureTypes.LABEL)
            else:
                feature_types.append(FeatureTypes.UNKNOWN)


        # Build value store.
        for patient_row in csv_reader:
            for value, feature, feature_type in zip(patient_row, cols, feature_types):
                # Replace '' with 'NA'.
                if value == '':
                    value = 'NA'
                if feature_type == FeatureTypes.ENUM and value not in vocab[feature]:
                    vocab[feature].add(value)

    with open(config.vocab_filename , mode='w') as vocab_file:
        for feature, values in vocab.items():
            line = [feature]
            for value in values:
                line.append(value)
            line.append('\n')
            vocab_file.write(','.join(line))

    return vocab



class WarfarinDataSet():

    def __init__(self, config):
        '''
        Args:
            config: an object of DataSetConfig.

        '''
        self.next = 0

        with open(config.data_filename, mode='r', newline='') as data_file:
            # Build feature types list.
            csv_reader = reader(data_file)
            self.cols = next(csv_reader)
            feature_types = []
            for col in self.cols:
                if col in config.enum_feature_cols:
                    feature_types.append(FeatureTypes.ENUM)
                elif col in config.float_feature_cols:
                    feature_types.append(FeatureTypes.FLOAT)
                elif col == config.label_col:
                    feature_types.append(FeatureTypes.LABEL)
                else:
                    feature_types.append(FeatureTypes.UNKNOWN)


            # Build enum feature vocabulary.
            # vocab: {feature: {val0, val1, val2...}, ...}
            self.vocab = {}
            with open(config.vocab_filename, 'r') as vocab_file:
                for line in vocab_file:
                    tokens = line.split(',')
                    self.vocab[tokens[0]] = {}
                    for ind, value in enumerate(tokens[1:-1]):
                        self.vocab[tokens[0]][value] = ind


            # Build value store.
            self.examples = []

            for patient_row in csv_reader:
                features = np.array([])
                skip = False
                for value, feature_type, col in zip(patient_row, feature_types, self.cols):
                    # Replace '' with 'NA'.
                    if value == '':
                        value = 'NA'
                    if col in config.required_features and value == 'NA':
                        skip = True
                        continue
                    if feature_type == FeatureTypes.ENUM:
                        enum_map = self.vocab[col]
                        feature = np.zeros([len(enum_map)], dtype=np.float32)
                        feature[enum_map[value]] = 1.0
                        features = np.concatenate((features, feature))
                    elif feature_type == FeatureTypes.FLOAT:
                        try:
                            value = float(value)
                        except ValueError:
                            value = .0
                        feature = np.asarray([value], dtype=np.float32)
                        features = np.concatenate((features, feature))
                    elif feature_type == FeatureTypes.LABEL:
                        try:
                            label = float(value)
                        except ValueError:
                            skip = True

                if not skip:
                    self.examples.append({'features': features, 'label': label})

    def shuffle(self):
        random.shuffle(self.examples)

    def size(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


if __name__ == '__main__':
    BuildEnumVocab(config.ConstConfig(), True)
    BuildEnumVocab(config.ClinicalDosingConfig(), True)
