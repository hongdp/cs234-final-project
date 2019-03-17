class ConstConfig():
    def __init__(self):
        self.data_filename = 'data/warfarin.csv'
        self.vocab_filename= 'data/warfarin_vocab.txt'
        self.enum_feature_cols = {}
        self.float_feature_cols = {}
        self.label_col = 'Therapeutic Dose of Warfarin'
        self.agent_name = 'const'

class ClinicalDosingConfig():
    def __init__(self):
        self.data_filename = 'data/warfarin.csv'
        self.vocab_filename= 'data/warfarin_clinical_vocab.txt'
        self.enum_feature_cols = {'Age', 'Race', 'Amiodarone (Cordarone)', 'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin'}
        self.float_feature_cols = {'Weight (kg)', 'Height (cm)'}
        self.label_col = 'Therapeutic Dose of Warfarin'
        self.required_features  = {'Age', 'Height (cm)', 'Weight (kg)'}
        self.agent_name = 'clinical_dosing'

class LinUCBConfig():
    def __init__(self):
        self.data_filename = 'data/warfarin.csv'
        self.vocab_filename= 'data/warfarin_linucb_vocab.txt'
        self.enum_feature_cols = {'Age', 'Race', 'Amiodarone (Cordarone)', 'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin', 'Was Dose of Acetaminophen or Paracetamol (Tylenol) >1300mg/day'}
        self.float_feature_cols = {'Weight (kg)', 'Height (cm)'}
        self.label_col = 'Therapeutic Dose of Warfarin'
        self.required_features  = {'Age', 'Height (cm)', 'Weight (kg)'}
        self.agent_name = 'LinUCB'

def get_config(agent_name):
    if agent_name == 'const':
        return ConstConfig()
    elif agent_name == 'clinical_dosing':
        return ClinicalDosingConfig()
    elif agent_name == 'LinUCB':
        return LinUCBConfig()