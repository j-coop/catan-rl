import gymnasium as gym
import numpy as np

from params.catan_constants import *
from params.tiles2nodes_adjacency_map import TILES_TO_NODES
from params.nodes2tiles_adjacency_map import NODES_TO_TILES

class CatanBaseEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.state = None
        self.observation_space = gym.spaces.Dict({
            "tiles": gym.spaces.Dict({
                "resources": gym.spaces.MultiBinary([N_TILES, 
                                                     N_RESOURCE_TYPES]),
                "tokens": gym.spaces.MultiBinary([N_TILES, 
                                                  N_TOKEN_VALUES]),
                "has_robber": gym.spaces.MultiBinary([N_TILES]),
                "nodes": gym.spaces.Dict({
                    "is_settlement": gym.spaces.MultiBinary([N_TILES, 6]),
                    "is_city":       gym.spaces.MultiBinary([N_TILES, 6]),
                    "owner":         gym.spaces.MultiBinary([N_TILES, 6,
                                                             N_PLAYERS]),
                    "ports":         gym.spaces.MultiBinary([N_TILES, 6,
                                                             N_PORT_FIELD_TYPES])
                }),
                "edges": gym.spaces.Dict({
                    "is_road": gym.spaces.MultiBinary([N_TILES, 6]),
                    "owner":   gym.spaces.MultiBinary([N_TILES, 6, N_PLAYERS]),
                })
            })
        })

    def reset(self):
        resources = self.__generate_resources()
        desert_tile_id = self.__get_desert_tile_id(resources)
        tokens = self.__generate_tokens(desert_tile_id)
        robber_index = desert_tile_id

        self.state = {
            "tiles": {
                "resources": resources,
                "tokens": tokens,
                "has_robber": np.eye(N_TILES)[robber_index],
                "nodes": {
                    "is_settlement": np.zeros((N_TILES, 6),
                                              dtype=np.int8),
                    "is_city": np.zeros((N_TILES, 6),
                                        dtype=np.int8),
                    "owner": np.zeros((N_TILES, 6, N_PLAYERS),
                                      dtype=np.int8),
                    "has_port": np.zeros((N_TILES, 6, N_PORT_FIELD_TYPES),
                                         dtype=np.int8),
                    "ports": self.__generate_ports()
                },
                "edges": {
                    "is_road": np.zeros((N_TILES, 6), dtype=np.int8),
                    "owner": np.zeros((N_TILES, 6, N_PLAYERS), dtype=np.int8),
                }
            }
        }
        return self.state

    def __generate_resources(self):
        """
        Return a MultiBinary matrix of shape (19, 6), one-hot per tile
        """
        res_list = (
            [0] * TILE_TYPE_COUNTS[RESOURCE_TYPES[0]] +
            [1] * TILE_TYPE_COUNTS[RESOURCE_TYPES[1]] +
            [2] * TILE_TYPE_COUNTS[RESOURCE_TYPES[2]] +
            [3] * TILE_TYPE_COUNTS[RESOURCE_TYPES[3]] +
            [4] * TILE_TYPE_COUNTS[RESOURCE_TYPES[4]] +
            [5] * TILE_TYPE_COUNTS[RESOURCE_TYPES[5]]
        )
        np.random.shuffle(res_list)
        return np.eye(N_RESOURCE_TYPES)[res_list].astype(np.int8)

    def __generate_tokens(self, desert_tile_id):
        tokens = np.zeros((N_TILES, N_TOKEN_VALUES), dtype=np.int8)

        shuffled_tokens = ALL_TOKENS.copy()
        np.random.shuffle(shuffled_tokens)

        token_idx = 0
        for i in range(N_TILES):
            if i == desert_tile_id:
                continue
            token_val = shuffled_tokens[token_idx]
            tokens[i][token_val - 2] = 1  # Map token 2–12 to index 0–10
            token_idx += 1
        return tokens

    def __generate_ports(self):
        res_list = (
            [0] * PORT_TYPE_COUNTS[PORT_TYPES[0]] +
            [1] * PORT_TYPE_COUNTS[PORT_TYPES[1]] +
            [2] * PORT_TYPE_COUNTS[PORT_TYPES[2]] +
            [3] * PORT_TYPE_COUNTS[PORT_TYPES[3]] +
            [4] * PORT_TYPE_COUNTS[PORT_TYPES[4]] +
            [5] * PORT_TYPE_COUNTS[PORT_TYPES[5]] +
            [6] * PORT_TYPE_COUNTS[PORT_TYPES[6]]
        )
        np.random.shuffle(res_list)
        node_ports = [val for val in res_list for _ in range(2)]
        node_ports = np.eye(N_PORT_FIELD_TYPES)[node_ports]
        result = np.zeros((N_NODES, N_PORT_FIELD_TYPES), dtype=np.int8)
        border_index = 0
        for node in range(N_NODES):
            if len(NODES_TO_TILES[node]) <= 2:
                result[node] = node_ports[border_index]
                border_index += 1

        tile_ports = np.zeros((N_TILES, 6, N_PORT_FIELD_TYPES), dtype=np.int8)
        for tile_id, node_ids in TILES_TO_NODES.items():
            for i, node_id in enumerate(node_ids):
                if len(NODES_TO_TILES[node]) <= 2:
                    tile_ports[tile_id, i] = result[node_id]
        return tile_ports

    def __get_desert_tile_id(self, resources):
        return np.argmax(resources[:, -1])
