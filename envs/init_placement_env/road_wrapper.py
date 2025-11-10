import gymnasium as gym
import numpy as np

from params.catan_constants import N_EDGES
from .env import CatanInitPlacementEnv


class CatanRoadPlacementEnv(gym.Env):

    def __init__(self, core_env: CatanInitPlacementEnv):
        super().__init__()
        self.core = core_env
        self.action_space = gym.spaces.Discrete(N_EDGES)
        self.observation_space = self.core.observation_space

    def get_action_masks(self):
        player = self.core._turn_order[self.core._turn_index]
        return self.core._road_placement_mask[:, player].copy()

    def reset(self, seed=None, options=None):
        # Don’t reset the shared environment
        return self.core._obs, {}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action format"
        action = np.eye(N_EDGES)[action]

        player = self.core._turn_order[self.core._turn_index]
        self.core._make_road_action(player, action)
        done = self.core._check_all_moves_done(action)

        reward = self.core._calculate_road_action_reward(action)
        self.core._after_road(done, player)
        return self.core._obs, reward, done, False, {}
