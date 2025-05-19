import gymnasium as gym
from gymnasium import spaces
import numpy as np

from params.catan_constants import *
from params.tiles2nodes_adjacency_map import TILES_TO_NODES
from params.nodes2nodes_adjacency_map import NODES_TO_NODES
from reset_mixins import CatanResetMixin

class CatanInitPlacementEnv(gym.Env,
                            CatanResetMixin):

    def __init__(self, base_env_obs):
        super().__init__()
        self.__base_obs = base_env_obs

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
                                                N_ADJACENT_NODES]),
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

        if settlement_action.sum() == 1:
            self.__make_settlement_action(settlement_action)
        elif road_action.sum() == 1:
            self.__make_road_action
        raise ValueError("Action must specify either a road or a settlement.")
    
    def __is_valid_settlement_placement(self, node_id):
        if self.__obs["adjacent_nodes"]["is_built"][node_id].any():
            return False  # adjacent node already has a settlement
        return True  # You can add more checks if needed

    def __apply_settlement(self, node_id):
        self.__obs["adjacent_nodes"]["is_built"][node_id] = 1
        self.__obs["adjacent_nodes"]["is_owned"][node_id] = 1

    def __is_valid_road_placement(self, edge_id):
        # Optional: Add checks for adjacency to a settlement
        return self.__obs["edges"]["is_built"][:, edge_id].sum() == 0

    def __apply_road(self, edge_id):
        self.__obs["edges"]["is_built"][:, edge_id] = 1
        self.__obs["edges"]["is_owned"][:, edge_id] = 1

    def __check_if_placement_done(self):
        # Return True if all agents have finished their initial placements
        # Could track self.__num_settlements or self.__placements_done
        return False
    
    def __verify_action(self, action, settlement_action, road_action):
        assert self.action_space.contains(action), "Invalid action format"
        assert (settlement_action.sum() + road_action.sum()) == 1, \
            "Exactly one action should be performed per step"
        
    def __make_settlement_action(self, settlement_action):
        node_id = np.argmax(settlement_action)
        if not self.__is_valid_settlement_placement(node_id):
            reward = -1.0
            terminated = True
            truncated = False
            return self.__obs, reward, terminated, truncated, {}

        self.__apply_settlement(node_id)
        self.__update_obs_after_settlement(node_id)
        reward = 1.0  # Or 0.0 if using sparse reward
        terminated = self.__check_if_placement_done()
        truncated = False
        return self.__obs, reward, terminated, truncated, {}

    def __make_road_action(self, road_action):
        edge_id = np.argmax(road_action)
        if not self.__is_valid_road_placement(edge_id):
            reward = -1.0
            terminated = True
            truncated = False
            return self.__obs, reward, terminated, truncated, {}

        self.__apply_road(edge_id)
        self.__update_obs_after_road(edge_id)
        reward = 1.0  # Or 0.0
        terminated = self.__check_if_placement_done()
        truncated = False
        return self.__obs, reward, terminated, truncated, {}

    def __prepare_obs_dict(self):
        obs = {
            "tiles": {
                "exist": np.zeros((N_NODES, N_ADJACENT_TILES), 
                                   dtype=np.int8),
                "tokens": np.zeros((N_NODES, N_ADJACENT_TILES, N_TOKEN_VALUES),
                                    dtype=np.int8),
                "resources": np.zeros((N_NODES, 
                                       N_ADJACENT_TILES, 
                                       N_RESOURCE_TYPES),
                                       dtype=np.int8),
            },
            "edges": {
                "exist": np.zeros((N_NODES, N_ADJACENT_EDGES),
                                dtype=np.int8),
                "is_built": np.zeros((N_NODES, N_ADJACENT_EDGES),
                                    dtype=np.int8),
                "is_owned": np.zeros((N_NODES, N_ADJACENT_EDGES),
                                    dtype=np.int8),
            },
            "adjacent_nodes": {
                "exist": np.zeros((N_NODES, N_ADJACENT_NODES),
                                dtype=np.int8),
                "is_built": np.zeros((N_NODES, N_ADJACENT_NODES),
                                    dtype=np.int8),
                "is_owned": np.zeros((N_NODES, N_ADJACENT_NODES),
                                    dtype=np.int8),
                "has_port": np.zeros((N_NODES,
                                      N_ADJACENT_NODES, 
                                      N_PORT_FIELD_TYPES),
                                      dtype=np.int8),
            },
            "has_port": np.zeros((N_NODES, N_PORT_FIELD_TYPES),
                                 dtype=np.int8),
        }
        return obs

    def __fill_tiles_info(self, tile_resources, tile_tokens):
        tile_resources = self.__base_obs["tiles"]["resources"]
        tile_tokens = self.__base_obs["tiles"]["tokens"]
        for tile_id, node_ids in TILES_TO_NODES.items():
            for i, node_id in enumerate(node_ids):
                self.__obs["tiles"]["exist"][node_id, i, 0] = 1
                self.__obs["tiles"]["resources"][node_id, i] = tile_resources[tile_id]
                self.__obs["tiles"]["tokens"][node_id, i] = tile_tokens[tile_id]

    def __fill_nodes_existence_info(self):
        for node_id in range(N_NODES):
            for i, neighbor in enumerate(self.__ring_neighbors[node_id]):
                if neighbor != -1:
                    self.__obs["adjacent_nodes"]["exist"][node_id, i] = 1


    def __fill_port_info(self):
        tile_ports = self.__base_obs["nodes"]["ports"]
        for tile_id, node_ids in enumerate(TILES_TO_NODES):
            for local_idx, node_id in enumerate(node_ids):
                port_vec = tile_ports[tile_id, local_idx]
                if port_vec.any():  # If there's a port
                    self.__obs["has_port"][node_id] = port_vec
                    for i, _ in enumerate(self.__ring_neighbors[node_id]):
                        self.__obs["adjacent_nodes"]["has_port"][node_id][i] = port_vec

    def __fill_indirect_edge_existence_info(self):
        for node_id in range(N_NODES):
            edge_set = set()
            direct_neighbors = NODES_TO_NODES[node_id]

            for neighbor in direct_neighbors:
                second_degree = NODES_TO_NODES.get(neighbor, [])
                for nn in second_degree:
                    if nn != node_id:
                        edge = tuple(sorted((neighbor, nn)))
                        edge_set.add(edge)

            for i, edge in enumerate(edge_set):
                self.__obs["edges"]["exist"][node_id, i] = 1


    def __find_neighbors_of_neighbors(self):
        max = 6
        result = np.full((N_NODES, max), -1, dtype=np.int32)

        for node in range(N_NODES):
            direct_neighbors = NODES_TO_NODES.get(node, [])
            second_hop_neighbors = set()

            for neighbor in direct_neighbors:
                second_hop_neighbors.update(NODES_TO_NODES.get(neighbor, []))

            # Remove the original node and direct neighbors from the result
            second_hop_neighbors.discard(node)
            second_hop_neighbors.difference_update(direct_neighbors)

            # Fill up to 6 values with -1 padding
            sorted_neighbors = sorted(second_hop_neighbors)
            result[node, :len(sorted_neighbors)] = sorted_neighbors[:max]
        self.__ring_neighbors = result
