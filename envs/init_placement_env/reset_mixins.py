import numpy as np

from params.catan_constants import *
from params.nodes2tiles_adjacency_map import NODES_TO_TILES
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

    def __generate_obs(self):
        self.__fill_tiles_info()
        self.__compute_ring_nodes()
        self.__compute_ring_edges()

        self.__fill_nodes_existence_info()
        self.__fill_edge_existence_info()
        self.__fill_port_info()

        return self._obs

    def __fill_tiles_info(self):
        tile_resources = self._base_obs["tiles"]["resources"]
        tile_tokens = self._base_obs["tiles"]["tokens"]
        for node_id, tile_ids in NODES_TO_TILES.items():
            print(node_id, tile_ids)
            for i, tile_id in enumerate(tile_ids):
                self._obs["tiles"]["exist"][node_id, i] = 1
                self._obs["tiles"]["resources"][node_id, i] = tile_resources[tile_id]
                self._obs["tiles"]["tokens"][node_id, i] = tile_tokens[tile_id]

    def __fill_nodes_existence_info(self):
        for node_id in range(N_NODES):
            for i, neighbor in enumerate(self.__ring_neighbors[node_id]):
                if neighbor != -1:
                    self._obs["adjacent_nodes"]["exist"][node_id, i] = 1

    def __fill_port_info(self):
        tile_ports = self._base_obs["tiles"]["nodes"]["ports"]
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

            # Fill up to N_ADJACENT_NODES values with -1 padding
            sorted_neighbors = sorted(second_hop_neighbors)
            n_neighbors = min(len(sorted_neighbors), N_ADJACENT_NODES)
            self.__ring_neighbors[node, :n_neighbors] = sorted_neighbors[:n_neighbors]
            if n_neighbors < N_ADJACENT_NODES:
                self.__ring_neighbors[node, n_neighbors:] = -1

    def __compute_ring_edges(self):
        """
        Compute and store adjacent node pairs for each node
        (indirect edge structure).
        For each node, store pairs of nodes that form edges in its ring
        (nodes that are 2 hops away, connected through a common neighbor).
        """
        for node_id in range(N_NODES):
            direct_neighbors = NODES_TO_NODES.get(node_id, [])
            edge_counter = 0

            for neighbor in direct_neighbors:
                # Get nodes that are neighbors of our neighbor (excluding our original node)
                second_degree_nodes = [n for n in NODES_TO_NODES.get(neighbor, []) if n != node_id]
                
                # For each pair of second-degree nodes that share our neighbor, create an edge in the ring
                for i in range(len(second_degree_nodes)):
                    for j in range(i + 1, len(second_degree_nodes)):
                        if edge_counter < N_ADJACENT_EDGES:
                            # Store the edge as a sorted pair to maintain consistency
                            a, b = sorted([second_degree_nodes[i], second_degree_nodes[j]])
                            self.__ring_edges[node_id][edge_counter] = [a, b]
                            edge_counter += 1
