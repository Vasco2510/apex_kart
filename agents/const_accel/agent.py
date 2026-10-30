"""Linea base sin aprendizaje: acelera recto, no hace nada mas."""
from stk_agent import Action


class Agent:
    def __init__(self, config):
        self.config = config

    def reset(self):
        pass

    def act(self, rgb):
        return Action(acceleration=1.0)

    def close(self):
        pass
