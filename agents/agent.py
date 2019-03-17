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

    def feedback(self, reward, context):
        raise NotImplementedError

