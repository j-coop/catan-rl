import gymnasium as gym
from gymnasium import spaces

from params.catan_constants import *


class CatanInitPlacementEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Dict({
            "build_road":       spaces.MultiBinary(N_EDGES),
            "build_settlement": spaces.MultiBinary(N_NODES),
        })

        self.observation_space = spaces.Dict({
            "tiles": spaces.Dict({
                "exist": spaces.MultiBinary([N_NODES,
                                             N_ADJACENT_TILES,
                                             1]),
                "tokens":     spaces.MultiBinary([N_NODES,
                                                  N_ADJACENT_TILES,
                                                  N_TOKEN_VALUES]),
                "resources":  spaces.MultiBinary([N_NODES,
                                                  N_ADJACENT_TILES,
                                                  N_RESOURCE_TYPES])
            }),
            "edges": spaces.Dict({
                "exist": spaces.MultiBinary([N_ADJACENT_EDGES, 1]),
                "is_built": spaces.MultiBinary([N_ADJACENT_EDGES, 1]),
                "is_owned": spaces.MultiBinary([N_ADJACENT_EDGES, 1])  # owned by the player
            }),
            "nodes": spaces.Dict({
                "exist": spaces.MultiBinary([N_ADJACENT_NODES, 1]),
                "is_built": spaces.MultiBinary([N_ADJACENT_NODES, 1]),
                "is_owned": spaces.MultiBinary([N_ADJACENT_NODES, 1]),  # owned by the player
                "is_port": spaces.MultiBinary([N_ADJACENT_NODES, 1])
            })
        })
