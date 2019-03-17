from .clinical_dosing_agent import ClinicalDosingAgent
from .const_agent import ConstAgent
from .LinUCB_agent import LinUCBAgent
from .agent import Actions

def get_agent(agent_name, config, vocab):
    if agent_name == 'const':
        return ConstAgent(config, vocab)
    elif agent_name  == 'clinical_dosing':
        return ClinicalDosingAgent(config, vocab)
    elif agent_name == 'LinUCB':
        return LinUCBAgent(config, vocab)