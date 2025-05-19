import gymnasium as gym
from gymnasium import spaces
import numpy as np

from params.catan_constants import *
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
        self.__settlement_mask = np.ones((N_NODES, N_PLAYERS), dtype=np.int8)
        self.__road_mask = np.ones((N_EDGES, N_PLAYERS), dtype=np.int8),

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
