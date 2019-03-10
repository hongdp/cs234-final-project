from enum import Enum
import config
import numpy as np

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

    with open(config.data_filename , 'r') as data_file:
        # Build feature types list.
        header_line = next(data_file)
        cols = header_line.split(',')
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
        for patient_row in data_file:
            for value, feature, feature_type in zip(patient_row.split(','), header_line.split(','), feature_types):
                if feature_type == FeatureTypes.ENUM and value not in vocab[feature]:
                    vocab[feature].add(value)
    with open(config.vocab_filename , 'w') as vocab_file:
        for feature, values in vocab.iteritems():
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

        with open(config.data_filename, 'r') as data_file:
            # Build feature types list.
            header_line = next(data_file)
            self.cols = header_line.split(',')
            print(self.cols)
            feature_types = []
            for col in self.cols:
                if col in config.enum_feature_cols:
                    feature_types.append(FeatureTypes.ENUM)
                elif col in config.float_feature_cols:
                    feature_types.append(FeatureTypes.FLOAT)
                elif col == config.label_col:
                    feature_types.append(FeatureTypes.LABEL)
                    print("label found")
                else:
                    feature_types.append(FeatureTypes.UNKNOWN)


            # Build enum feature vocabulary.
            # vocab: {feature: {val0, val1, val2...}, ...}
            self.vocab = {}
            with open(config.vocab_filename, 'r') as vocab_file:
                for line in vocab_file:
                    tokens = line.split(',')
                    self.vocab[tokens[0]] = {}
                    for ind, value in enumerate(tokens[1:]):
                        self.vocab[tokens[0]][value] = ind
            print(self.vocab)


            # Build value store.
            self.examples = []
            nolabelcnt = 0

            for patient_row in data_file:
                features = np.array([])
                label_found = True
                for value, feature_type, feature in zip(patient_row.split(','), feature_types, self.cols):
                    # print("value: {} feature: {} type: {}".format(value, feature, feature_type))
                    if feature_type == FeatureTypes.ENUM:
                        enum_map = self.vocab[feature]
                        feature = np.zeros([len(enum_map)], dtype=np.float32)
                        feature[enum_map[value]] = 1.0
                        features = np.concatenate((features, feature))
                    elif feature_type == FeatureTypes.FLOAT:
                        try:
                            value = float(value)
                        except ValueError:
                            value = .0
                            # print("cant convert to float: {}".format(value))
                        feature = np.asarray([value], dtype=np.float32)
                        features = np.concatenate((features, feature))
                    elif feature_type == FeatureTypes.LABEL:
                        try:
                            label = float(value)
                        except ValueError:
                            # print("no label: {} cnt: {}".format(value, nolabelcnt))
                            nolabelcnt += 1
                            label_found = False

                if label_found:
                    self.examples.append({'features': features, 'label': label})

            print(self.examples)

    def __iter__(self):
        return self

    def next():
        '''
        Return raw feature and label of next patient in the form of:
            {'features': np.array(np.float32), 'label':np.float32}
        '''
        self.next += 1
        if self.next > len(self.values):
            raise StopIteration
        return self.values[self.next-1]


if __name__ == '__main__':
    BuildEnumVocab(config.DataSetConfig(), True)
    BuildEnumVocab(config.ClinicalDataSetConfig(), True)
