import gymnasium as gym
import random

from params.catan_constants import (N_NODES,
                                    N_EPISODES)
from params.edges_list import EDGES_LIST
from .env import CatanInitPlacementEnv


class CatanSettlementPlacementEnv(gym.Env):

    def __init__(self, core_env: CatanInitPlacementEnv):
        super().__init__()
        self.core = core_env
        self.action_space = gym.spaces.Discrete(N_NODES)
        self.observation_space = self.core.observation_space

    def get_action_masks(self):
        return self.core._settlement_placement_mask.copy()

    def reset(self, seed=None, options=None):
        obs, _ = self.core.reset(seed=seed)
        return obs, {}
    
    def get_random_road_action(self, node):
        # Filter all edges that include the node
        matching = [edge for edge in EDGES_LIST if node in edge]
        if not matching:
            raise ValueError(f"No edges found that include node {node}")
        return random.choice(matching)

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action format"

        turn = self.core._turn_index
        player = self.core._turn_order[turn]
        self.core._make_settlement_action(player, action)

        reward = self.core._calculate_settlement_action_reward(action)
        self.core._after_settlement(reward)
        self.core._make_road_action(player, self.get_random_road_action(action))
        done = True if self.core._check_all_moves_done() else False
        if done:
            self.core._episode_counter += 1
            print(f'{self.core._episode_counter} / {N_EPISODES}')
        return self.core._obs, reward, done, False, {}
