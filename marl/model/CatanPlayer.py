from collections import Counter
from typing import Dict, List

import numpy as np

from marl.model.CatanBoard import CatanBoard
from marl.params.catan_constants import RESOURCE_TYPES, DEV_CARD_TYPES, PORT_TYPES, BUILD_COSTS


class CatanPlayer:
    """
    Stores player-specific game state.
    """
    def __init__(self, color: str):
        self.color: str = color
        self.resources: Dict[str, int] = {res: 0 for res in RESOURCE_TYPES}
        self.dev_cards: Dict[str, int] = {card: 0 for card in DEV_CARD_TYPES}
        self.ports: Dict[str, bool] = {port: False for port in PORT_TYPES}
        self.settlements: List[int] = []  # node indices
        self.cities: List[int] = []       # node indices
        self.roads: List[int] = []        # edge indices
        self.points: int = 0

    def can_afford(self, build_type: str) -> bool:
        """
        Check if the player can afford a given build (settlement, city, road, dev_card),
        considering resources + trades (bank 4:1, ports 3:1 or 2:1).
        """
        cost = Counter(BUILD_COSTS[build_type])
        available = Counter(self.resources)

        # Direct check
        if all(available[res] >= qty for res, qty in cost.items()):
            return True

        # Compute shortages
        shortages = {res: max(0, qty - available[res]) for res, qty in cost.items()}
        total_needed = sum(shortages.values())

        # Check if trades can cover shortages
        tradable = 0
        for res, qty in available.items():
            if res in shortages:
                continue  # don’t trade away resources you already need
            if qty <= 0:
                continue

            # Determine trade ratio
            if self.ports.get(res, False):
                # specific 2:1 port
                tradable += qty // 2
            elif self.ports["3for1"]:
                # generic 3:1 port
                tradable += qty // 3
            else:
                # bank 4:1
                tradable += qty // 4

        return tradable >= total_needed


    def can_afford_settlement(self) -> bool:
        return self.can_afford("settlement")

    def can_afford_city(self) -> bool:
        return self.can_afford("city")

    def can_afford_road(self) -> bool:
        return self.can_afford("road")

    def can_afford_dev_card(self) -> bool:
        return self.can_afford("dev_card")

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