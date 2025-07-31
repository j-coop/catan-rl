from pettingzoo import AECEnv


class CatanEnv(AECEnv):

    metadata = {"name": "catan_v0"}

    def __init__(self):
        super().__init__()
        self.agents = ['blue', 'red', 'white', 'black']
        self.possible_agents = self.agents[:]
        self.agent_selection = None

    def step(self, action):
        pass

    def reset(self, seed = None, options = None):
        pass

    def observe(self, agent):
        pass

    def render(self):
        pass

    def state(self):
        pass
