from enum import Enum

class FeatureTypes(Enum):
    IGNORE = 0 # Feature will be ignored.
    ENUM = 1 # Feature value will be treated as enum values and turned into one hot representation.
    FLOAT = 2 # Feature value will be directly treated as float values.

def BuildEnumVocab(config, output=False):
    '''
    Build vocabulary for enum features requested in DataSetConfig.
    The vocabulary will be dumped to the vocab file specified in config if output is ture.
    Return:
        vocab: {feature: {val0, val1, val2...}, ...}
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
            else:
                feature_types.append(FeatureTypes.IGNORE)


        # Build value store.
        for patient_row in data_file:
            for value, feature, feature_type in zip(patient_row.split(','), header_line.split(','), feature_types):
                if feature_type == FeatureTypes.ENUM and value not in vocab[feature]:
                    vocab[feature].add(value)
    print(vocab)
    with open(config.vocab_filename , 'w') as vocab_file:
        for feature, values in vocab.iteritems():
            line = [feature]
            for value in values:
                line.append(value)
            line.append('\n')
            vocab_file.write(','.join(line))

    return vocab


class DataSetConfig():
    def __init__(self):
        self.data_filename = 'data/warfarin.csv'
        self.vocab_filename= 'data/warfarin_vocab.txt'
        self.enum_feature_cols = {}
        self.float_feature_cols = {}
        self.label_col = ['Therapeutic Dose of Warfarin']

class ClinicalDataSetConfig(DataSetConfig):
    def __init__(self):
        DataSetConfig.__init__(self)
        self.data_filename = 'data/warfarin.csv'
        self.vocab_filename= 'data/warfarin_clinical_vocab.txt'
        self.enum_feature_cols = {'Race', 'Amiodarone (Cordarone)', 'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampici'}
        self.float_feature_cols = {}
        self.label_col = ['Therapeutic Dose of Warfarin']

class WarfarinDataSet():

    def __init__(self, config):
        '''
        Args:
            config: an object of DataSetConfig.

        '''
        self.next = 0

        with open(config.filename, 'r') as data_file:
            # Build feature types list.
            header_line = next(data_file)
            self.cols = header_line.split(',')
            print(cols)
            feature_types = []
            for col in self.cols:
                if col in config.enum_feature_cols:
                    feature_types.append(FeatureTypes.ENUM)
                elif col in config.float_feature_cols:
                    feature_types.append(FeatureTypes.FLOAT)
                else:
                    feature_types.append(FeatureTypes.IGNORE)


            # Build enum feature vocabulary.

            # Build value store.
            self.values = []
            for patient_row in data_file:
                for value, feature_type in zip(header_line.split(','), feature_types):
                    if feature_type == FeatureTypes.ENUM:
                        pass

            print(self.values)

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

    def LoadVocab(self, vocal_file):
        pass

if __name__ == '__main__':
    BuildEnumVocab(DataSetConfig(), True)
    BuildEnumVocab(ClinicalDataSetConfig(), True)
