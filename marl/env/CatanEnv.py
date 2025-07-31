from gymnasium import spaces
from pettingzoo import AECEnv

from marl.params.catan_constants import N_NODES, N_EDGES


class CatanEnv(AECEnv):

    metadata = {"name": "catan_v0"}

    def __init__(self):
        super().__init__()
        self.agents = ['blue', 'red', 'white', 'black']
        self.possible_agents = self.agents[:]
        self.agent_selection = None

        self.action_spaces = {
            agent: spaces.Discrete(self.get_action_space_size()) for agent in self.agents
        }

    def get_action_space_size(self) -> int:
        size = 0
        # N_NODES for placing settlements and cities each
        size += 2 * N_NODES
        # N_EDGES for placing roads
        size += N_EDGES
        # 1 for buy dev card
        # 5 for playing dev cards
        # 1 for move robber (steal included)
        # 1 for end turn
        size += 1 + 5 + 1 + 1
        return size


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
