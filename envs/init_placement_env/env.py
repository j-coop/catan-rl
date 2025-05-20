import gymnasium as gym
from gymnasium import spaces
import numpy as np

from params.nodes2nodes_adjacency_map import NODES_TO_NODES
from params.catan_constants import *
from params.edges_list import EDGES_LIST
from reset_mixins import CatanResetMixin
from step_mixins import CatanStepMixin

class CatanInitPlacementEnv(CatanResetMixin,
                            CatanStepMixin,
                            gym.Env):

    def __init__(self, base_env_obs):
        super().__init__()
        self.__base_obs = base_env_obs
        self.__turn_order = [0, 1, 2, 3, 3, 2, 1, 0]
        self.__turn_index = 0
        self.__placement_stage = "settlement"
        self.__settlement_placement_mask = np.ones((N_NODES), dtype=np.int8)
        self.__road_placement_mask = np.zeros((N_EDGES, N_PLAYERS),
                                             dtype=np.int8)
        self._obs = self.__prepare_obs_dict()

        self.action_space = spaces.Dict({
            "build_settlement": spaces.MultiBinary(N_NODES),
            "build_road":       spaces.MultiBinary(N_EDGES),
        })

        self.observation_space = spaces.Dict({
            "tiles": spaces.Dict({
                "exist":      spaces.MultiBinary([N_NODES,
                                                  N_ADJACENT_TILES]),
                "tokens":     spaces.MultiBinary([N_NODES,
                                                  N_ADJACENT_TILES,
                                                  N_TOKEN_VALUES]),
                "resources":  spaces.MultiBinary([N_NODES,
                                                  N_ADJACENT_TILES,
                                                  N_RESOURCE_TYPES])
            }),
            "edges": spaces.Dict({
                "exist":      spaces.MultiBinary([N_NODES,
                                                  N_ADJACENT_EDGES]),
                "is_built":   spaces.MultiBinary([N_NODES,
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
            "has_port": np.zeros((N_NODES, N_PORT_FIELD_TYPES),
                        dtype=np.int8),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._obs = self.__prepare_obs_dict()
        self.__fill_tiles_info()
        self.__find_neighbors_of_neighbors()
        self.__fill_nodes_existence_info()
        self.__fill_port_info()
        self.__fill_indirect_edge_existence_info()
        self.observation_space = self._obs
        del self._obs

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
        Settlements are rewarded after full episode (16 steps) - late reward after rolls simulation
        """
        is_road = self.__is_placing_1_road(road_action)

        if self.__is_placing_1_settlement(settlement_action):
            self.__make_settlement_action(player, settlement_action)
        elif self.__is_placing_1_road(road_action):
            self.__make_road_action(player, road_action)
        # raise ValueError("Action must specify either 1 road or 1 settlement.")

        if is_road:
            reward = 0 # TODO: road reward heuristic
        else:
            reward = 0 # no immediate reward for settlement

        done = self._turn_index == len(self._turn_order) - 1
        self._turn_index += 1

        return self._obs, reward, done, False, {}



    def __update_settlement_placement_mask(self, node_id):
        """
        Disable settlement placement for all players on the given node
        and its adjacent nodes.
        """
        affected_nodes = [node_id] + NODES_TO_NODES.get(node_id, [])
        for n in affected_nodes:
            self.__settlement_placement_mask[n] = 0  # Disable for all agents

    def __update_road_placement_mask(self, settled_node: int, agent_id: int):
        for neighbor in NODES_TO_NODES[settled_node]:
            edge = tuple(sorted((settled_node, neighbor)))
            try:
                edge_index = EDGES_LIST.index(edge)
                self.__road_placement_mask[edge_index, agent_id] = 1
            except ValueError:
                raise ValueError(f"Edge {edge} not found in EDGES_LIST.")
