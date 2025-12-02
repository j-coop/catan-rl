import random
import numpy as np

from params.catan_constants import (N_EDGES,
                                    N_NODES,
                                    INIT_PLACEMENT_ENV_N_EPISODES)
from params.edges_list import EDGES_LIST
from .env import CatanInitPlacementEnv


class CatanSettlementPlacementEnv(CatanInitPlacementEnv):

    def __init__(self,
                 ep_done_previously=0,
                 base_env_obs=None,
                 train=True):
        super().__init__(ep_done_previously, base_env_obs, train)

    def get_action_masks(self):
        s_mask = self._settlement_placement_mask.astype(bool)
        r_mask = np.zeros((N_EDGES,), dtype=bool)
        return np.concatenate([s_mask, r_mask])

    def _get_random_road_action(self, node):
        # Filter all edges that include the node
        matching = [edge for edge in EDGES_LIST if node in edge]
        if not matching:
            raise ValueError(f"No edges found that include node {node}")
        return random.choice(matching)

    def step(self, action):
        assert action < N_NODES, f"Got road action {action}, masking failed"
        action = int(action)
        turn = self._turn_index
        player = self._turn_order[turn]
        self._make_settlement_action(player, action)

        reward = self._calculate_settlement_action_reward(action)
        self._after_settlement(reward)
        if self._train:
            self._make_road_action(player,
                                   self._get_random_road_action(action))
        done = True if self._check_all_moves_done() else False
        if done:
            self._episode_counter += 1
            ep_number = self._ep_done_previously + self._episode_counter
            print(f'{ep_number} / {INIT_PLACEMENT_ENV_N_EPISODES}')
        return self._obs, reward, done, False, {'base_obs': self._base_obs}
