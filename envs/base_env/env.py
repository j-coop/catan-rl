import gym
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
                    "is_city": gym.spaces.MultiBinary([N_TILES, 6]),
                    "owner": gym.spaces.MultiBinary([N_TILES, 6, N_PLAYERS]),
                    "ports": gym.spaces.MultiBinary([N_TILES, 6,
                                                     N_PORT_FIELD_TYPES])
                })
            })
        })

    def reset(self):
        resources = self.__generate_resources()
        tokens = self.__generate_tokens(resources)
        robber_index = np.argmax(resources[:, -1])  # Desert tile index

        self.state = {
            "tiles": {
                "resources": resources,
                "tokens": tokens,
                "has_robber": np.eye(self.N_TILES)[robber_index],
                "nodes": {
                    "is_settlement": np.zeros((self.N_TILES, 6),
                                              dtype=np.int8),
                    "is_city": np.zeros((self.N_TILES, 6),
                                        dtype=np.int8),
                    "owner": np.zeros((self.N_TILES, 6, N_PLAYERS),
                                        dtype=np.int8),
                    "has_port": np.zeros((N_TILES, 6, N_PORT_FIELD_TYPES),
                                         dtype=np.int8),
                    "ports": self.__generate_ports()
                }
            },
        }
        return self.state

    def __generate_resources(self):
        """
        Return a MultiBinary matrix of shape (19, 6), one-hot per tile
        """
        res_list = (
            [0] * TILE_TYPE_COUNTS["brick"] +
            [1] * TILE_TYPE_COUNTS["wood"] +
            [2] * TILE_TYPE_COUNTS["wool"] +
            [3] * TILE_TYPE_COUNTS["grain"] +
            [4] * TILE_TYPE_COUNTS["ore"] +
            [5] * TILE_TYPE_COUNTS["desert"]
        )
        np.random.shuffle(res_list)
        return np.eye(N_RESOURCE_TYPES)[res_list]

    def __generate_tokens(self):
        token_values = ALL_TOKENS
        np.random.shuffle(token_values)
        tokens = np.zeros((N_TILES, N_TOKEN_VALUES), dtype=np.int32)

        token_idx = 0
        desert_tile_index = self.__get_desert_tile_index()
        for i in range(N_TILES):
            if i == desert_tile_index:
                continue
            token_val = token_values[token_idx]
            tokens[i][token_val - 2] = 1  # map 2-12 to index 0-10
            token_idx += 1
        return tokens

    def __generate_ports(self):
        res_list = (
            [0] * PORT_TYPE_COUNTS["brick"] +
            [1] * PORT_TYPE_COUNTS["wood"] +
            [2] * PORT_TYPE_COUNTS["wool"] +
            [3] * PORT_TYPE_COUNTS["grain"] +
            [4] * PORT_TYPE_COUNTS["ore"] +
            [5] * PORT_TYPE_COUNTS["generic"] +
            [6] * PORT_TYPE_COUNTS["no_port"]
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
        for tile_id, node_ids in enumerate(TILES_TO_NODES):
            for i, node_id in enumerate(node_ids):
                if len(NODES_TO_TILES[node]) <= 2:
                    tile_ports[tile_id, i] = result[node_id]
        return tile_ports

    def __get_desert_tile_index(self):
        resources = self.state["tiles"]["resources"]
        return np.argmax(resources[:, -1])
