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

class LassoConfig():
    def __init__(self):
        self.data_filename = 'data/warfarin.csv'
        self.vocab_filename= 'data/warfarin_lasso_vocab.txt'
        # self.enum_feature_cols = {'Age', 'Race', 'Amiodarone (Cordarone)', 'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin'}
        self.enum_feature_cols = {'Age', 'Race', 'Amiodarone (Cordarone)', 'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin', 'Cyp2C9 genotypes', 'Genotyped QC Cyp2C9*2', 'Genotyped QC Cyp2C9*3', 'Combined QC CYP2C9', 'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', 'VKORC1 QC genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', 'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C', 'VKORC1 QC genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C', 'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G', 'VKORC1 QC genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G', 'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G', 'VKORC1 QC genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G', 'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G', 'VKORC1 QC genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G', 'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G', 'VKORC1 QC genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G', 'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C', 'VKORC1 QC genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C', 'CYP2C9 consensus', 'VKORC1 -1639 consensus', 'VKORC1 497 consensus', 'VKORC1 1173 consensus', 'VKORC1 1542 consensus', 'VKORC1 3730 consensus', 'VKORC1 2255 consensus', 'VKORC1 -4451 consensus'}
        self.float_feature_cols = {'Weight (kg)', 'Height (cm)'}
        self.label_col = 'Therapeutic Dose of Warfarin'
        self.required_features  = {'Age', 'Height (cm)', 'Weight (kg)'}
        self.agent_name = 'lasso'
        self.lasso_q = 1
        self.lasso_h = 5
        self.lasso_lam1 = 5e-2
        self.lasso_lam2 = 5e-2

class LinUCBConfig():
    def __init__(self):
        self.data_filename = 'data/warfarin.csv'
        self.vocab_filename= 'data/warfarin_linucb_vocab.txt'
        # self.enum_feature_cols = {'Age', 'Race', 'Amiodarone (Cordarone)', 'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin'}
        self.enum_feature_cols = {'Age', 'Race', 'Amiodarone (Cordarone)', 'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin', 'Cyp2C9 genotypes', 'Genotyped QC Cyp2C9*2', 'Genotyped QC Cyp2C9*3', 'Combined QC CYP2C9', 'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', 'VKORC1 QC genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', 'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C', 'VKORC1 QC genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C', 'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G', 'VKORC1 QC genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G', 'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G', 'VKORC1 QC genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G', 'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G', 'VKORC1 QC genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G', 'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G', 'VKORC1 QC genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G', 'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C', 'VKORC1 QC genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C', 'CYP2C9 consensus', 'VKORC1 -1639 consensus', 'VKORC1 497 consensus', 'VKORC1 1173 consensus', 'VKORC1 1542 consensus', 'VKORC1 3730 consensus', 'VKORC1 2255 consensus', 'VKORC1 -4451 consensus'}
        self.float_feature_cols = {'Weight (kg)', 'Height (cm)'}
        self.label_col = 'Therapeutic Dose of Warfarin'
        self.required_features  = {'Age', 'Height (cm)', 'Weight (kg)'}
        self.agent_name = 'LinUCB'


def get_config(agent_name):
    configs = [ConstConfig(),ClinicalDosingConfig(),LassoConfig(),LinUCBConfig()]
    for config in configs:
        if config.agent_name == agent_name:
            return config
    raise ValueError('No config found.')

