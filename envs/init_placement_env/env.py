from math import floor

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from params.nodes2nodes_adjacency_map import NODES_TO_NODES
from params.catan_constants import *
from params.edges_list import EDGES_LIST
from .reset_mixins import CatanResetMixin
from .step_mixins import CatanStepMixin
from .validation_mixin import CatanValidationMixin


class CatanInitPlacementEnv(CatanResetMixin,
                            CatanStepMixin,
                            CatanValidationMixin,
                            gym.Env):

    def __init__(self, base_env_obs):
        gym.Env.__init__(self)
        CatanResetMixin.__init__(self)
        self._base_obs = base_env_obs
        self._turn_order = [0, 1, 2, 3, 3, 2, 1, 0]
        self._turn_index = 0
        self._placement_stage = "settlement"
        self.__settlement_placement_mask = np.ones((N_NODES, ), dtype=np.int8)
        self.__road_placement_mask = np.zeros((N_EDGES, N_PLAYERS),
                                             dtype=np.int8)
        self._last_settlement_node_index = 0
        self._settlement_gains = np.zeros((N_PLAYERS, 2, N_RESOURCE_TYPES))

        self.__step_counter = 0

        """
        Flat action space (some frameworks and algorithms don't work with dicts in action space)
        First N_NODES bits for settlement actions
        Further N_EDGES bits for road actions
        """
        self.action_space = spaces.Discrete(N_NODES + N_EDGES)

        self.observation_space = spaces.Dict({
            "tiles_exist": spaces.MultiBinary([N_NODES, N_ADJACENT_TILES]),
            "tiles_tokens": spaces.MultiBinary([N_NODES, N_ADJACENT_TILES, N_TOKEN_VALUES]),
            "tiles_resources": spaces.MultiBinary([N_NODES, N_ADJACENT_TILES, N_RESOURCE_TYPES]),
            "edges_exist": spaces.MultiBinary([N_NODES, N_ADJACENT_EDGES]),
            "edges_is_built": spaces.MultiBinary([N_NODES, N_ADJACENT_EDGES]),
            "adj_exist": spaces.MultiBinary([N_NODES, N_ADJACENT_NODES]),
            "adj_is_built": spaces.MultiBinary([N_NODES, N_ADJACENT_NODES]),
            "adj_has_port": spaces.MultiBinary([N_NODES, N_ADJACENT_NODES, N_PORT_FIELD_TYPES]),
            "has_port": spaces.MultiBinary([N_NODES, N_PORT_FIELD_TYPES])
        })

        self._obs = self._CatanResetMixin__prepare_obs_dict()

    def get_action_masks(self) -> np.ndarray:
        print(self._turn_index)
        player = self._turn_order[self._turn_index]

        # Start with base masks
        settlement_mask = self.__settlement_placement_mask.copy()
        road_mask = self.__road_placement_mask[:, player].copy()

        # Mask out actions depending on current placement stage
        if self._placement_stage == 'road':
            settlement_mask[:] = 0  # Disable all settlement actions
        elif self._placement_stage == 'settlement':
            road_mask[:] = 0  # Disable all road actions

        # Concatenate into flat action mask
        action_mask = np.concatenate([settlement_mask, road_mask])
        return action_mask

    def reset(self, seed=None, options=None):
        print('=================== RESET ===================')
        super().reset(seed=seed)
        self._obs = self._CatanResetMixin__prepare_obs_dict()

        self._obs = self._CatanResetMixin__generate_obs()

        self._turn_index = 0
        self._placement_stage = "settlement"
        self.__settlement_placement_mask = np.ones((N_NODES,), dtype=np.int8)
        self.__road_placement_mask = np.zeros((N_EDGES, N_PLAYERS),
                                              dtype=np.int8)
        self._last_settlement_node_index = 0
        self._settlement_gains = np.zeros((N_PLAYERS, 2, N_RESOURCE_TYPES))

        return self._obs, {}

    def step(self, action):
        """
        Agent places EITHER a settlement OR a road in this step.
        """
        print(f'STEP {self.__step_counter}')
        print(action)
        # Convert discrete action to one-hot encoded values
        settlement_action = np.zeros(N_NODES, dtype=np.int8)
        road_action = np.zeros(N_EDGES, dtype=np.int8)
        
        if action < N_NODES:
            # Settlement
            settlement_action[action] = 1
            print(f'    PLACING SETTLEMENT')
        else:
            # Road
            road_action[action - N_NODES] = 1
            print('    PLACING ROAD')

        self._verify_action(action, settlement_action, road_action)
        player = self._turn_order[self._turn_index]
        print(f'    PLAYER {player}')

        """
        Road gets instant heuristic reward for step
        Settlements are rewarded based on simulated resources gained
        Reward is given for number of simulated resources gained for both settlements
        Additional reward after second settlement for resources diversity
        """
        is_road = self._is_placing_1_road(road_action)
        print(f'is_road: {is_road}')

        if self._is_placing_1_settlement(settlement_action):
            print('    PLACING SETTLEMENT')
            self._make_settlement_action(player, settlement_action)
        elif self._is_placing_1_road(road_action):
            print('    PLACING ROAD')
            self._make_road_action(player, road_action)
        else:
            raise ValueError("Action must specify either 1 road or 1 settlement.")

        # Set rewards
        if is_road:
            reward = self._evaluate_road_heuristic(road_action, self._last_settlement_node_index)
            reward += REWARD_WEIGHTS["ROAD"]
        else:
            reward = self._simulate_dice_rolls(settlement_action)
            reward += REWARD_WEIGHTS["RESOURCES_NUM"]

        done = self._turn_index == len(self._turn_order) - 1 and is_road
        if not is_road:
            # Settlement
            self._last_settlement_node_index = np.argmax(settlement_action)
            self._placement_stage = "road"
            is_second_settlement = floor((self._turn_index + 1) / 4)
            if is_second_settlement:
                # Final reward for resources diversity
                normalized_reward = self._evaluate_final_resources(self._settlement_gains)
                reward += normalized_reward * REWARD_WEIGHTS["RESOURCES_DISTRIBUTION"]
        else:
            # Road
            if not done:
                self._turn_index += 1
            else:
                self._turn_index = 0
            self._placement_stage = "settlement"
            # Set road masks to zero to avoid building second road from the same settlement
            self.__road_placement_mask[:, player] = 0

        self.__step_counter += 1
        if self.__step_counter >= 16:
            self.__step_counter = 0

        return self._obs, reward, done, False, {'base_obs': self._base_obs}

    def _update_settlement_placement_mask(self, node_id):
        """
        Disable settlement placement for all players on the given node
        and its adjacent nodes.
        """
        affected_nodes = [node_id] + NODES_TO_NODES.get(node_id, [])
        for n in affected_nodes:
            self.__settlement_placement_mask[n] = 0  # Disable for all agents

    def _update_road_placement_mask(self, settled_node: int, player_id: int):
        print('_update_road_placement_mask')
        print(settled_node)
        print(player_id)
        print(NODES_TO_NODES[settled_node])
        for neighbor in NODES_TO_NODES[settled_node]:
            edge = tuple(sorted((settled_node, neighbor)))
            try:
                edge_index = EDGES_LIST.index(edge)
                self.__road_placement_mask[edge_index, player_id] = 1
            except ValueError:
                raise ValueError(f"Edge {edge} not found in EDGES_LIST.")
