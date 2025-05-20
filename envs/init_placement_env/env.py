import gymnasium as gym
from gymnasium import spaces
import numpy as np

from params.nodes2nodes_adjacency_map import NODES_TO_NODES
from params.catan_constants import *
from params.edges_list import EDGES_LIST
from reset_mixins import CatanResetMixin
from step_mixins import CatanStepMixin

class CatanInitPlacementEnv(gym.Env,
                            CatanResetMixin,
                            CatanStepMixin):

    def __init__(self, base_env_obs):
        super().__init__()
        self.__base_obs = base_env_obs
        self.__turn_order = [0, 1, 2, 3, 3, 2, 1, 0]
        self.__turn_index = 0
        self.__placement_stage = "settlement"
        self.__settlement_placement_mask = np.ones((N_NODES), dtype=np.int8)
        self.__settlement_ownership_mask = np.zeros((N_NODES, N_PLAYERS), 
                                                    dtype=np.int8)
        self.__road_placement_mask = np.zeros((N_EDGES, N_PLAYERS),
                                             dtype=np.int8)
        self.__road_ownership_mask = np.zeros((N_EDGES, N_PLAYERS),
                                             dtype=np.int8)

        self.action_space = spaces.Dict({
            "build_settlement": spaces.MultiBinary(N_NODES),
            "build_road":       spaces.MultiBinary(N_EDGES),
        })

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
                "is_owned": spaces.MultiBinary([N_NODES,  # owned by the player
                                                N_ADJACENT_EDGES]),
            }),
            "adjacent_nodes": spaces.Dict({
                "exist": spaces.MultiBinary([N_NODES, 
                                             N_ADJACENT_NODES]),
                "is_built": spaces.MultiBinary([N_NODES, 
                                                N_ADJACENT_NODES]),
                "is_owned": spaces.MultiBinary([N_NODES, # owned by the player
                                                N_ADJACENT_NODES,
                                                N_PLAYERS]),
                "is_port": spaces.MultiBinary([N_NODES, 
                                               N_ADJACENT_NODES]),
            })
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.__obs = self.__prepare_obs_dict()
        self.__fill_tiles_info()
        self.__find_neighbors_of_neighbors()
        self.__fill_nodes_existence_info()
        self.__fill_port_info()
        self.__fill_indirect_edge_existence_info()
        self.observation_space = self.__obs
        del self.__obs

    def step(self, action):
        """
        Agent places EITHER a settlement OR a road in this step.
        """
        settlement_action = action["build_settlement"]
        road_action = action["build_road"]
        self.__verify_action(settlement_action, road_action)
        agent = self.__turn_order[self.__turn_index]

        if self.__is_placing_1_settlement(settlement_action):
            self.__make_settlement_action(agent, settlement_action)
        elif self.__is_placing_1_road(road_action):
            self.__make_road_action(agent, road_action)
        raise ValueError("Action must specify either 1 road or 1 settlement.")

    def __update_settlement_placement_mask(self, node_id):
        """
        Disable settlement placement for all players on the given node
        and its adjacent nodes.
        """
        affected_nodes = [node_id] + NODES_TO_NODES.get(node_id, [])
        for n in affected_nodes:
            self.__settlement_placement_mask[n] = 0  # Disable for all agents

    def __update_settlement_ownership_mask(self, node_id, agent_id):
        """
        Set ownership of the node by particular agent
        """
        affected_nodes = [node_id] + NODES_TO_NODES.get(node_id, [])
        for n in affected_nodes:
            self.__settlement_ownership_mask[n, agent_id] = 1

    def __update_road_ownership_mask(self,
                                     edge_nodes: tuple[int, int], 
                                     agent_id: int):
        # Ensure the edge is always in sorted order
        sorted_edge = tuple(sorted(edge_nodes))
        try:
            edge_index = EDGES_LIST.index(sorted_edge)
            self.__road_ownership_mask[edge_index, agent_id] = 1
        except ValueError:
            raise ValueError(f"Edge {sorted_edge} not found in EDGES_LIST.")

    def __update_road_placement_mask(self, settled_node: int, agent_id: int):
        for neighbor in NODES_TO_NODES[settled_node]:
            edge = tuple(sorted((settled_node, neighbor)))
            try:
                edge_index = EDGES_LIST.index(edge)
                self.__road_placement_mask[edge_index, agent_id] = 1
            except ValueError:
                raise ValueError(f"Edge {edge} not found in EDGES_LIST.")
