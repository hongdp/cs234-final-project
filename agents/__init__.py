from .clinical_dosing_agent import ClinicalDosingAgent
from .const_agent import ConstAgent
from .lasso_agent import LassoAgent
from .agent import Actions

def get_agent(agent_name, config, vocab):
    if agent_name == 'const':
        return ConstAgent(config, vocab)
    elif agent_name  == 'clinical_dosing':
        return ClinicalDosingAgent(config, vocab)
    elif agent_name  == 'lasso':
        return LassoAgent(config, vocab)
