import numpy as np

from params.catan_constants import *
from params.tiles2nodes_adjacency_map import TILES_TO_NODES
from params.nodes2nodes_adjacency_map import NODES_TO_NODES


class CatanResetMixin:

    def __init__(self):
        self.__ring_edges = np.zeros((N_NODES, N_ADJACENT_EDGES, 2),
                                     dtype=np.uint8)
        self.__ring_neighbors = np.full((N_NODES, N_ADJACENT_NODES), -1, dtype=np.int8)

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
            },
            "adjacent_nodes": {
                "exist": np.zeros((N_NODES, N_ADJACENT_NODES),
                                   dtype=np.int8),
                "is_built": np.zeros((N_NODES, N_ADJACENT_NODES),
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
        tile_resources = self._base_obs["tiles"]["resources"]
        tile_tokens = self._base_obs["tiles"]["tokens"]
        for tile_id, node_ids in TILES_TO_NODES.items():
            for i, node_id in enumerate(node_ids):
                self._obs["tiles"]["exist"][node_id, i, 0] = 1
                self._obs["tiles"]["resources"][node_id, i] = tile_resources[tile_id]
                self._obs["tiles"]["tokens"][node_id, i] = tile_tokens[tile_id]

    def __fill_nodes_existence_info(self):
        for node_id in range(N_NODES):
            for i, neighbor in enumerate(self.__ring_neighbors[node_id]):
                if neighbor != -1:
                    self._obs["adjacent_nodes"]["exist"][node_id, i] = 1

    def __fill_port_info(self):
        tile_ports = self._base_obs["nodes"]["ports"]
        for tile_id, node_ids in TILES_TO_NODES.items():
            for local_idx, node_id in enumerate(node_ids):
                port_vec = tile_ports[tile_id, local_idx]
                if port_vec.any():  # If there's a port
                    self._obs["has_port"][node_id] = port_vec
                    for i, _ in enumerate(self.__ring_neighbors[node_id]):
                        self._obs["adjacent_nodes"]["has_port"][node_id][i] = port_vec

    def __fill_edge_existence_info(self):
        """Mark edges as existing based on previously computed ring edges."""
        self._obs["edges"]["exist"] = np.zeros((N_NODES, N_ADJACENT_EDGES),
                                                dtype=np.int8)
        for node_id in range(N_NODES):
            for edge_id in range(N_ADJACENT_EDGES):
                a, b = self.__ring_edges[node_id][edge_id]
                if a != 0 or b != 0:  # assumes empty = [0, 0]
                    self._obs["edges"]["exist"][node_id, edge_id] = 1


    def __compute_ring_nodes(self):
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
            self.__ring_neighbors[node, :len(sorted_neighbors)] = sorted_neighbors[:max]

    def __compute_ring_edges(self):
        """
        Compute and store adjacent node pairs for each node
        (indirect edge structure).
        """
        for node_id in range(N_NODES):
            direct_neighbors = NODES_TO_NODES[node_id]
            edge_counter = 0

            for neighbor in direct_neighbors:
                second_degree_nodes = NODES_TO_NODES.get(neighbor, []).copy()
                second_degree_nodes.remove(neighbor)
                if len(second_degree_nodes) == 1:
                    self.__ring_edges[node_id][edge_counter][0] = neighbor
                    self.__ring_edges[node_id][edge_counter][1] = second_degree_nodes[0]
                    edge_counter += 1 if second_degree_nodes[0] < neighbor else 2
                else:
                    for i in range(2):
                        self.__ring_edges[node_id][edge_counter][0] = neighbor
                        self.__ring_edges[node_id][edge_counter][1] = second_degree_nodes[i]
                        edge_counter += 1
