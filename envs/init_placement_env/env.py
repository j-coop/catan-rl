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
        self._settlement_gains = np.zeros((N_PLAYERS, 2))

        """
        Flat action space (some frameworks and algorithms don't work with dicts in action space)
        First N_NODES bits for settlement actions
        Further N_EDGES bits for road actions
        """
        self.action_space = spaces.MultiBinary(N_NODES + N_EDGES)

        self.observation_space = spaces.Dict({
            "tiles": spaces.Dict({
                "exist": spaces.MultiBinary([N_NODES,
                                             N_ADJACENT_TILES]),
                "tokens":     spaces.MultiBinary([N_NODES,
                                                  N_ADJACENT_TILES,
                                                  N_TOKEN_VALUES]),
                "resources":  spaces.MultiBinary([N_NODES,
                                                  N_ADJACENT_TILES,
                                                  N_RESOURCE_TYPES])
            }),
            "edges": spaces.Dict({
                "exist": spaces.MultiBinary([N_NODES,
                                             N_ADJACENT_EDGES]),
                "is_built": spaces.MultiBinary([N_NODES,
                                                N_ADJACENT_EDGES]),
            }),
            "adjacent_nodes": spaces.Dict({
                "exist":      spaces.MultiBinary([N_NODES,
                                                  N_ADJACENT_NODES]),
                "is_built":   spaces.MultiBinary([N_NODES,
                                                  N_ADJACENT_NODES]),
                "has_port":   spaces.MultiBinary([N_NODES,
                                                  N_ADJACENT_NODES,
                                                  N_PORT_FIELD_TYPES]),
            }),
            "has_port": spaces.MultiBinary((N_NODES, N_PORT_FIELD_TYPES)),
        })

        self._obs = self._CatanResetMixin__prepare_obs_dict()

    def get_action_masks(self) -> np.ndarray:
        player = self._turn_order[self._turn_index]

        # Mask for settlements (length N_NODES)
        settlement_mask = self.__settlement_placement_mask
        # Mask for roads for the current player (length N_EDGES)
        road_mask = self.__road_placement_mask[:, player]

        # Concatenate the two into one flat mask
        action_mask = np.concatenate([settlement_mask, road_mask])  # shape: (N_NODES + N_EDGES,)

        return action_mask


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._obs = self._CatanResetMixin__prepare_obs_dict()

        self._obs = self._CatanResetMixin__generate_obs()

        return self._obs

    def step(self, action):
        """
        Agent places EITHER a settlement OR a road in this step.
        """
        settlement_action = action["build_settlement"]
        road_action = action["build_road"]
        self.__verify_action(action, settlement_action, road_action)
        player = self._turn_order[self._turn_index]

        """
        Road gets instant heuristic reward for step
        Settlements are rewarded based on simulated resources gained
        Reward is given for number of simulated resources gained for both settlements
        Additional reward after second settlement for resources diversity
        """
        is_road = self.__is_placing_1_road(road_action)

        if self.__is_placing_1_settlement(settlement_action):
            self.__make_settlement_action(player, settlement_action)
        elif self.__is_placing_1_road(road_action):
            self.__make_road_action(player, road_action)
        else:
            raise ValueError("Action must specify either 1 road or 1 settlement.")

        # Set rewards
        if is_road:
            reward = self.__evaluate_road_heuristic(road_action, self._last_settlement_node_index)
            reward += REWARD_WEIGHTS["ROAD"]
        else:
            reward = self.__simulate_dice_rolls(settlement_action)
            reward += REWARD_WEIGHTS["RESOURCES_NUM"]

        done = self._turn_index == len(self._turn_order) - 1
        if not is_road:
            self._last_settlement_node_index = np.argmax(settlement_action)
            self._placement_stage = "road"
            is_second_settlement = floor((self._turn_index + 1) / 4)
            if is_second_settlement:
                # Final reward for resources diversity
                normalized_reward = self.__evaluate_final_resources(self._settlement_gains)
                reward += normalized_reward * REWARD_WEIGHTS["RESOURCES_DISTRIBUTION"]
        else:
            self._turn_index += 1
            self._placement_stage = "settlement"

        return self.observation_space, reward, done, False, {}

    def _update_settlement_placement_mask(self, node_id):
        """
        Disable settlement placement for all players on the given node
        and its adjacent nodes.
        """
        affected_nodes = [node_id] + NODES_TO_NODES.get(node_id, [])
        for n in affected_nodes:
            self.__settlement_placement_mask[n] = 0  # Disable for all agents

    def _update_road_placement_mask(self, settled_node: int, player_id: int):
        for neighbor in NODES_TO_NODES[settled_node]:
            edge = tuple(sorted((settled_node, neighbor)))
            try:
                edge_index = EDGES_LIST.index(edge)
                self.__road_placement_mask[edge_index, player_id] = 1
            except ValueError:
                raise ValueError(f"Edge {edge} not found in EDGES_LIST.")
