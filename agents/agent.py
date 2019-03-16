from enum import Enum


class Actions(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2

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
