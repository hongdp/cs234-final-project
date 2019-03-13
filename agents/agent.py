class Agent():
    def __init__(self, vocab):
        raise NotImplementedError

    def act(self, feature):
        raise NotImplementedError

def get_agent(agent_name):
    if env_name == 'const':
        return ConstConfig()
    elif env_name == 'clinical_dosing':
        return ClinicalDosingConfig()
