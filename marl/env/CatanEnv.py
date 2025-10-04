import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from marl.model.CatanGame import CatanGame
from marl.params.catan_constants import N_NODES, N_EDGES


class CatanEnv(AECEnv):

    metadata = {"name": "catan_v0"}

    def __init__(self):
        super().__init__()
        self.agents = ['blue', 'red', 'white', 'black']
        self.possible_agents = self.agents[:]
        self.agent_selection = self.agents[0]

        # Game Logic Layer object
        self.game = CatanGame(self.agents)

        self.action_spaces = {
            agent: spaces.Discrete(self.get_action_space_size()) for agent in self.agents
        }

        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(self.action_spaces[agent].n,), dtype=np.int8
                    ),
                    "observation": spaces.Box(
                        low=0, high=1, shape=(self.get_observation_space_size(),), dtype=np.float32
                    ),
                }
            )
            for agent in self.agents
        }

    @staticmethod
    def get_action_space_size() -> int:
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

    @staticmethod
    def get_observation_space_size() -> int:
        size = 0
        # TODO: define observation space shape
        return size


    def step(self, action):
        agent = self.agent_selection
        player = self.game.current_player

        # Apply the action in the game logic
        self.apply_action(agent, action)

        # Compute rewards
        reward = self.compute_reward(agent)
        self.rewards[agent] = reward

        # Determine next agent
        self.agent_selection = self._next_agent()

        # Generate observation for next agent
        obs = self.observe(self.agent_selection)
        return obs, reward, self.terminations[agent], self.truncations[agent], {}

    def reset(self, seed = None, options = None):
        pass

    def observe(self, agent):
        pass

    def render(self):
        pass

    def state(self):
        pass
