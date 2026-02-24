import numpy as np

from params.catan_constants import (N_NODES,
                                    INIT_PLACEMENT_ENV_N_EPISODES)
from .env import CatanInitPlacementEnv


class CatanRoadPlacementEnv(CatanInitPlacementEnv):

    def __init__(self,
                 ep_done_previously=0,
                 base_env_obs=None,
                 train=True,
                 evaluation=False):
        super().__init__(ep_done_previously, base_env_obs, train)
        self._evaluation = evaluation

    def get_action_masks(self):
        s_mask = np.zeros((N_NODES,), dtype=bool)
        player = self.turn_order[self.turn_index]
        r_mask = self._road_placement_mask[:, player].astype(bool)
        return np.concatenate([s_mask, r_mask])

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action format"
        action = int(action) - N_NODES
        turn = self.turn_index
        player = self.turn_order[turn]

        self._make_road_action(player, action)
        reward = self._calculate_road_action_reward(action)
        self._after_road(player)
        done = self._check_all_moves_done()
        if done:
            self._episode_counter += 1
            ep_number = self._ep_done_previously + self._episode_counter
            print(f'{ep_number} / {INIT_PLACEMENT_ENV_N_EPISODES}')
        if self._train and not done:
            node_id = self._get_random_settlement_action()
            next_player = self.turn_order[self.turn_index]
            self._make_settlement_action(next_player, node_id)
        return self._obs, reward, done, False, {'base_obs': self._base_obs}

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if not self._evaluation:
            first_settlement_id = self._get_random_settlement_action()
            self._make_settlement_action(player=0, node_id=first_settlement_id)
        return self._obs, info
