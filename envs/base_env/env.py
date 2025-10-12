import gymnasium as gym
import numpy as np
from random import randint
import json
import uuid
import os

from params.catan_constants import *
from params.tiles2nodes_adjacency_map import TILES_TO_NODES
from params.nodes2tiles_adjacency_map import NODES_TO_TILES
from params.coastal_nodes_list import COASTAL_NODES_LIST

class CatanBaseEnv(gym.Env):

    def __init__(self, save_env=False):
        super().__init__()
        self.state = None
        self.save_env = save_env
        self.observation_space = gym.spaces.Dict({
            "resources": gym.spaces.MultiBinary([N_TILES, 
                                                    N_RESOURCE_TYPES]),
            "tokens": gym.spaces.MultiBinary([N_TILES, 
                                                N_TOKEN_VALUES]),
            "has_robber": gym.spaces.MultiBinary([N_TILES]),
            "nodes_settlements": gym.spaces.MultiBinary([N_TILES, 6]),
            "nodes_cities": gym.spaces.MultiBinary([N_TILES, 6]),
            "nodes_owners": gym.spaces.MultiBinary([N_TILES, 6,
                                                            N_PLAYERS]),
            "nodes_ports": gym.spaces.MultiBinary([N_TILES, 6,
                                                        N_PORT_TYPES]),
            "edges_owners": gym.spaces.MultiBinary([N_TILES, 6, N_PLAYERS]),
            "edges_roads": gym.spaces.MultiBinary([N_TILES, 6])
        })

    def reset(self):
        resources = self.__generate_resources()
        desert_tile_id = self.__get_desert_tile_id(resources)
        tokens = self.__generate_tokens(desert_tile_id)
        robber_index = desert_tile_id

        self.state = {
            "resources": resources,
            "tokens": tokens,
            "has_robber": np.eye(N_TILES)[robber_index],

            "nodes_settlements": np.zeros((N_TILES, 6), dtype=np.int8),
            "nodes_cities": np.zeros((N_TILES, 6), dtype=np.int8),
            "nodes_owners": np.zeros((N_TILES, 6, N_PLAYERS), dtype=np.int8),
            "nodes_ports": self.__generate_ports(),
            "edges_owners": np.zeros((N_TILES, 6, N_PLAYERS), dtype=np.int8),
            "edges_roads": np.zeros((N_TILES, 6), dtype=np.int8)
        }

        # Save generated env to json
        if self.save_env:
            # Create saves directory if it doesn't exist
            os.makedirs('saves', exist_ok=True)
            
            # Generate random UUID for filename
            filename = f"saves/catan_base_env_{uuid.uuid4()}.json"
            
            # Convert numpy arrays to compact format for JSON serialization
            state_dict = {}
            for key, value in self.state.items():
                if isinstance(value, np.ndarray):
                    # Convert to list and remove unnecessary nesting for 1D arrays
                    arr = value.tolist()
                    if len(value.shape) == 1:
                        state_dict[key] = arr
                    else:
                        # For multi-dimensional arrays, flatten and include shape
                        state_dict[key] = {
                            "data": [x for sublist in arr for x in (sublist if isinstance(sublist, list) else [sublist])],
                            "shape": value.shape
                        }
                else:
                    state_dict[key] = value
            
            # Save to JSON file in compact format
            with open(filename, 'w') as f:
                json.dump(state_dict, f, separators=(',', ':'))

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
    
    def __shuffle_coastal_nodes(self, i):
        coastal_nodes = COASTAL_NODES_LIST[i:] + COASTAL_NODES_LIST[:i]

    def __generate_ports(self):
        port_types = (
            [0] * PORT_TYPE_COUNTS[PORT_TYPES[0]] +
            [1] * PORT_TYPE_COUNTS[PORT_TYPES[1]] +
            [2] * PORT_TYPE_COUNTS[PORT_TYPES[2]] +
            [3] * PORT_TYPE_COUNTS[PORT_TYPES[3]] +
            [4] * PORT_TYPE_COUNTS[PORT_TYPES[4]] +
            [5] * PORT_TYPE_COUNTS[PORT_TYPES[5]]
        )
        np.random.shuffle(port_types)
        port_nodes = []
        for port_type in port_types:
            port_nodes.extend([-1, -1, port_type, port_type])
        port_nodes.extend([-1, -1])
        result = np.zeros((N_NODES, N_PORT_TYPES), dtype=np.int8)
        rnd = randint(0, 3)
        coastal_nodes = COASTAL_NODES_LIST[rnd:] + COASTAL_NODES_LIST[:rnd]
        for i, node_id in enumerate(coastal_nodes):
            if port_nodes[i] != -1:
                result[node_id] = np.eye(N_PORT_TYPES,
                                         dtype=np.int8)[port_nodes[i]]

        tile_ports = np.zeros((N_TILES, 6, N_PORT_TYPES), dtype=np.int8)
        for tile_id, node_ids in TILES_TO_NODES.items():
            for i, node_id in enumerate(node_ids):
                if node_id in COASTAL_NODES_LIST:
                    tile_ports[tile_id, i] = result[node_id]
        return tile_ports

    def __get_desert_tile_id(self, resources):
        return np.argmax(resources[:, -1])
