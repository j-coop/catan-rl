from typing import List, Optional

import numpy as np

from marl.model.CatanPlayer import CatanPlayer
from marl.params.catan_constants import N_NODES, N_EDGES


class CatanBoard:
    """
    Stores tiles, nodes, edges, and robber position.
    """
    def __init__(self):
        self.tiles: List[Optional[str]] = [None] * 19  # resource types
        self.nodes: List[Optional[str]] = [None] * N_NODES  # player name who owns settlement/city
        self.edges: List[Optional[str]] = [None] * N_EDGES  # player name who owns road
        self.robber_position: int = 0  # index of hex with robber

    def place_settlement(self, node: int, player: CatanPlayer):
        self.nodes[node] = player.color

    def place_road(self, edge: int, player: CatanPlayer):
        self.edges[edge] = player.color

    def get_board_observation(self) -> np.ndarray:
        # encoding: 0 if empty, 1-4 for player index ownership
        node_obs = np.array([0 if n is None else 1 for n in self.nodes], dtype=np.float32)
        edge_obs = np.array([0 if e is None else 1 for e in self.edges], dtype=np.float32)
        return np.concatenate([node_obs, edge_obs, [self.robber_position / len(self.tiles)]])
