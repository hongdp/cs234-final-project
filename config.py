class DataSetConfig():
    def __init__(self):
        self.data_filename = 'data/warfarin.csv'
        self.vocab_filename= 'data/warfarin_vocab.txt'
        self.enum_feature_cols = {}
        self.float_feature_cols = {}
        self.label_col = 'Therapeutic Dose of Warfarin'

class ClinicalDataSetConfig(DataSetConfig):
    def __init__(self):
        DataSetConfig.__init__(self)
        self.data_filename = 'data/warfarin.csv'
        self.vocab_filename= 'data/warfarin_clinical_vocab.txt'
        self.enum_feature_cols = {'Age', 'Race', 'Amiodarone (Cordarone)', 'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin'}
        self.float_feature_cols = {'Weight (kg)', 'Height (cm)'}
        self.label_col = 'Therapeutic Dose of Warfarin'
