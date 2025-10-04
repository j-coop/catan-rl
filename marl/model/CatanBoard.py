import random
from typing import List, Optional, Tuple

import numpy as np

from marl.model.CatanPlayer import CatanPlayer
from marl.params.catan_constants import N_NODES, N_EDGES

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
        self.tiles: List[Optional[Tuple[str, Optional[int]]]] = [None] * 19  # resource types
        self.nodes: List[Optional[str]] = [None] * N_NODES  # player who owns settlement/city
        self.edges: List[Optional[str]] = [None] * N_EDGES  # player who owns road
        self.robber_position: int = 0  # index of tile with robber

        self.generate_tiles()

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

    def place_settlement(self, node: int, player: CatanPlayer):
        self.nodes[node] = player.color

    def place_road(self, edge: int, player: CatanPlayer):
        self.edges[edge] = player.color

    def get_board_observation(self) -> np.ndarray:
        # encoding: 0 if empty, 1-4 for player index ownership
        node_obs = np.array([0 if n is None else 1 for n in self.nodes], dtype=np.float32)
        edge_obs = np.array([0 if e is None else 1 for e in self.edges], dtype=np.float32)
        return np.concatenate([node_obs, edge_obs, [self.robber_position / len(self.tiles)]])
