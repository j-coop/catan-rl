from typing import Dict, List

import numpy as np

from marl.model.CatanBoard import CatanBoard
from marl.params.catan_constants import RESOURCE_TYPES, DEV_CARD_TYPES


class CatanPlayer:
    """
    Stores player-specific game state.
    """
    def __init__(self, color: str):
        self.color: str = color
        self.resources: Dict[str, int] = {res: 0 for res in RESOURCE_TYPES}
        self.dev_cards: Dict[str, int] = {card: 0 for card in DEV_CARD_TYPES}
        self.settlements: List[int] = []  # node indices
        self.cities: List[int] = []       # node indices
        self.roads: List[int] = []        # edge indices
        self.points: int = 0

    def can_place_settlement(self, node: int, board: CatanBoard) -> bool:
        # Implement adjacency, distance rule, and resource checks
        return True

    def place_settlement(self, node: int, board: CatanBoard):
        self.settlements.append(node)
        # Deduct resources, update points, update board
        board.place_settlement(node, self)

    def get_observation(self) -> np.ndarray:
        # resources + number of settlements/cities/roads/victory points
        obs = np.array(
            list(self.resources.values())
            + [len(self.settlements), len(self.cities), len(self.roads), self.points],
            dtype=np.float32,
        )
        return obs