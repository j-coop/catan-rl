from math import floor
import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from params.nodes2nodes_adjacency_map import NODES_TO_NODES
from params.catan_constants import *
from params.edges_list import EDGES_LIST
from .reset_mixins import CatanResetMixin
from .step_mixins import CatanStepMixin
from .validation_mixin import CatanValidationMixin
from ..base_env.env import CatanBaseEnv


class CatanInitPlacementEnv(CatanResetMixin,
                            CatanStepMixin,
                            CatanValidationMixin,
                            gym.Env):

    def __init__(self,
                 ep_done_previously,
                 base_env_obs,
                 train):
        gym.Env.__init__(self)
        CatanResetMixin.__init__(self)
        self._base_obs = base_env_obs
        if not self._base_obs:
            base_env = CatanBaseEnv(save_env=False)
            self._base_obs = base_env.reset()
        self._train = train
        self.turn_order = [0, 1, 2, 3, 3, 2, 1, 0]
        self._turn_index = 0
        self._ep_done_previously = ep_done_previously
        self._settlement_placement_mask = np.ones((N_NODES,), dtype=np.int8)
        self._road_placement_mask = np.zeros((N_EDGES, N_PLAYERS), dtype=np.int8)
        self._last_settlement_node_index = 0
        self._settlement_gains = np.zeros((N_PLAYERS, 2, N_TILE_TYPES))
        self._episode_counter = 0
        self.action_space = gym.spaces.Discrete(N_NODES + N_EDGES)

        """
        Flat action space (some frameworks and algorithms don't work with dicts in action space)
        First N_NODES bits for settlement actions
        Further N_EDGES bits for road actions
        """

        self.observation_space = spaces.Dict({
            "tiles_exist": spaces.MultiBinary([N_NODES, N_ADJACENT_TILES]),
            "tiles_tokens": spaces.MultiBinary([N_NODES, N_ADJACENT_TILES, N_TOKEN_VALUES]),
            "tiles_resources": spaces.MultiBinary([N_NODES, N_ADJACENT_TILES, N_TILE_TYPES]),
            "adj_exist": spaces.MultiBinary([N_NODES, N_ADJACENT_NODES]),
            "adj_is_built": spaces.MultiBinary([N_NODES, N_ADJACENT_NODES]),
            "adj_has_port": spaces.MultiBinary([N_NODES, N_ADJACENT_NODES, N_PORT_TYPES]),
            "has_port": spaces.MultiBinary([N_NODES, N_PORT_TYPES])
        })
        self._obs = self._CatanResetMixin__prepare_obs_dict()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._train:
            base_env = CatanBaseEnv(save_env=False)
            self._base_obs = base_env.reset()

        self._obs = self._CatanResetMixin__prepare_obs_dict()
        self._obs = self._CatanResetMixin__generate_obs()

        self._turn_index = 0
        self._last_settlement_node_index = 0
        self._settlement_placement_mask = np.ones((N_NODES,), dtype=bool)
        self._road_placement_mask = np.zeros((N_EDGES, N_PLAYERS), dtype=bool)
        self._settlement_gains = np.zeros((N_PLAYERS, 2, N_TILE_TYPES))

        if self._train:
            first_settlement_id = self._get_random_settlement_action()
            self._make_settlement_action(player=0, node_id=first_settlement_id)

        return self._obs, {}

    def _calculate_significance_weight(self):
        alpha = 0.5 # exponential decay rate for early steps
        min_significance = 0.06 # minimum weight for last steps
        significance = max(alpha ** (self._turn_index - 1), min_significance)
        return significance

    def _calculate_road_action_reward(self, road_action):
        reward = self._evaluate_road_heuristic(road_action)
        return reward * REWARD_WEIGHTS["ROAD"]

    def _calculate_settlement_action_reward(self, settlement_action):

        def calculate_placement_reward(significance):
            placement_gain = self._evaluate_placement(settlement_action)
            return placement_gain * significance * REWARD_WEIGHTS["PLACEMENT"]

        def calculate_resource_reward(significance):
            res_gain = self._evaluate_expected_resource_gain(settlement_action)
            reward = res_gain - BASELINE_REWARD # [-0.43 ; 0.57]
            reward *= 2.0 # roughly [-1 ; 1] - considered best for PPO
            return reward * significance * REWARD_WEIGHTS["RESOURCES_NUM"]

        reward = 0
        significance = self._calculate_significance_weight()
        reward += calculate_resource_reward(significance)
        reward += calculate_placement_reward(significance)
        return reward

    def _check_all_moves_done(self):
        return self._turn_index == 0

    def _after_settlement(self, reward):
        is_second_settlement = floor((self._turn_index + 1) / 4)
        if is_second_settlement:
            norm_reward = self._evaluate_resources_distribution(self._settlement_gains)
            reward += norm_reward * REWARD_WEIGHTS["RESOURCES_DISTRIBUTION"]
        self._update_turn_index()

    def _after_road(self, player):
        self._road_placement_mask[:, player] = 0
        self._update_turn_index()

    def _update_turn_index(self):
        self._turn_index += 1
        self._turn_index %= len(self._turn_order)

    def _update_settlement_placement_mask(self, node_id):
        affected_nodes = [node_id] + NODES_TO_NODES.get(node_id, [])
        for n in affected_nodes:
            self._settlement_placement_mask[n] = 0  # Disable for all agents

    def _update_road_placement_mask(self, settled_node: int, player_id: int):
        for neighbor in NODES_TO_NODES[settled_node]:
            edge = tuple(sorted((settled_node, neighbor)))
            try:
                edge_index = EDGES_LIST.index(edge)
                self._road_placement_mask[edge_index, player_id] = 1
            except ValueError:
                raise ValueError(f"Edge {edge} not found in EDGES_LIST.")

    def _get_random_settlement_action(self):
        valid_indices = np.where(self._settlement_placement_mask == 1)[0]
        if valid_indices.size == 0:
            raise ValueError(f"No place for the settlement")
        return random.choice(valid_indices)
