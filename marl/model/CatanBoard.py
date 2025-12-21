from __future__ import annotations
import random
from typing import List, Optional, Tuple
import numpy as np

from marl.model.CatanPlayer import CatanPlayer
from params.catan_constants import N_NODES, N_EDGES, PORT_NODE_PAIRS
from params.edges_list import EDGES_LIST
from params.nodes2nodes_adjacency_map import NODES_TO_NODES

RESOURCE_DISTRIBUTION = [
    "wood", "wood", "wood", "wood",
    "brick", "brick", "brick",
    "sheep", "sheep", "sheep", "sheep",
    "wheat", "wheat", "wheat", "wheat",
    "ore", "ore", "ore",
    "desert"
]

TOKEN_DISTRIBUTION = [
    2, 12,
    3, 3,
    4, 4,
    5, 5,
    6, 6,
    8, 8,
    9, 9,
    10, 10,
    11, 11,
]


class CatanBoard:
    """
    Stores tiles, nodes, edges, and robber position.
    """
    def __init__(self):
        self.tiles: List[Tuple[str, Optional[int]]] = [None] * 19  # resource types
        self.nodes: List[Optional[str]] = [None] * N_NODES  # player who owns settlement/city
        self.edges: List[Optional[str]] = [None] * N_EDGES  # player who owns road
        self.ports: List[Optional[str]] = [None] * N_NODES  # type of port on each node
        self.robber_position: int = 0  # index of tile with robber

        self.generate_tiles()
        self.generate_ports()
        print(self.ports)

    def generate_tiles(self, seed: Optional[int] = None):
        """
        Generate a randomized board setup: resource tiles + number tokens.
        Desert gets no token and robber starts there.
        """
        rng = random.Random(seed)

        # Shuffle resources
        resources = RESOURCE_DISTRIBUTION[:]
        rng.shuffle(resources)

        # Shuffle tokens
        tokens = TOKEN_DISTRIBUTION[:]
        rng.shuffle(tokens)

        # Assign tokens to non-desert tiles
        tiles: List[Tuple[str, Optional[int]]] = []
        token_index = 0
        robber_position = None

        for i, res in enumerate(resources):
            if res == "desert":
                tiles.append((res, None))
                robber_position = i
            else:
                tiles.append((res, tokens[token_index]))
                token_index += 1

        # Save results
        self.tiles = tiles
        self.robber_position = robber_position if robber_position is not None else 0

    def generate_ports(self, seed: Optional[int] = None):
        """
        Assign port types to specific node indices around the board perimeter.
        Each port is associated with 2 adjacent nodes.
        """
        rand = random.Random(seed)

        # Standard Catan ports: 9 ports (5 resource-specific + 4 generic 3:1)
        port_types = ["wood", "brick", "wheat", "sheep", "ore"] + ["3for1"] * 4
        rand.shuffle(port_types)

        # Assign port types to these node pairs
        for port_type, (n1, n2) in zip(port_types, PORT_NODE_PAIRS):
            self.ports[n1] = port_type
            self.ports[n2] = port_type

    def place_settlement(self, node: int, player: CatanPlayer):
        self.nodes[node] = player.name

    def place_road(self, edge: int, player: CatanPlayer):
        self.edges[edge] = player.name

    def get_board_observation(self) -> np.ndarray:
        # encoding: 0 if empty, 1-4 for player index ownership
        node_obs = np.array([0 if n is None else 1 for n in self.nodes], dtype=np.float32)
        edge_obs = np.array([0 if e is None else 1 for e in self.edges], dtype=np.float32)
        return np.concatenate([node_obs, edge_obs, [self.robber_position / len(self.tiles)]])

    def get_valid_settlement_spots(self,
                                   player: CatanPlayer,
                                   init_placement=False) -> List[int]:
        """
        Returns all node indices where the player can legally build a settlement.
        - Node must be empty
        - Adjacent nodes must also be empty (distance rule)
        - Must be connected to player's road
        """
        valid_nodes = []

        for node_idx, owner in enumerate(self.nodes):
            # Node must be empty
            if owner is not None:
                continue

            # No adjacent settlements (distance rule)
            if any(self.nodes[adj] is not None for adj in NODES_TO_NODES.get(node_idx, [])):
                continue

            # Must connect to one of player’s roads
            if not init_placement:
                connected_edges = [edge_idx for edge_idx, (a, b) in enumerate(EDGES_LIST) if node_idx in (a, b)]
                if not any(self.edges[e] == player.name for e in connected_edges):
                    continue

            valid_nodes.append(node_idx)

        return valid_nodes

    def get_valid_road_spots(self, player: CatanPlayer) -> List[int]:
        """
        Returns all edge indices where the player can legally build a road.
        - Edge must be empty.
        - Must connect to player's existing road or settlement.
        """
        valid_edges = []

        for edge_idx, (a, b) in enumerate(EDGES_LIST):
            # Edge must be empty
            if self.edges[edge_idx] is not None:
                continue

            # Must connect to player’s structure (settlement/city) or road
            node_connection = self.nodes[a] == player.name or self.nodes[b] == player.name
            road_connection = any(
                self.edges[e] == player.name
                for e in self._get_connected_edges(a) + self._get_connected_edges(b)
            )
            if node_connection or road_connection:
                valid_edges.append(edge_idx)

        return valid_edges

    # --- Helper Methods ---

    @staticmethod
    def _get_connected_edges(node_idx: int) -> List[int]:
        """Return all edge indices that connect to the given node."""
        return [i for i, (a, b) in enumerate(EDGES_LIST) if node_idx in (a, b)]
