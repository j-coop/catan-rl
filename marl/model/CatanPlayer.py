from collections import Counter
from typing import Dict, List

import numpy as np

from marl.model.CatanBoard import CatanBoard
from params.nodes2tiles_adjacency_map import NODES_TO_TILES
from params.catan_constants import (DEV_CARD_TYPES, PORT_TYPES,
                                    RESOURCE_TYPES, BUILD_COSTS)


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

    def take_resources(self, roll: int, board: CatanBoard):
        """
        Collect resources for this player after a dice roll.
        Settlements yield +1, cities yield +2.
        Robber blocks production on its tile.
        """
        if roll == 7:
            return  # robber event, no production

        for node in self.settlements + self.cities:
            for tile_index in NODES_TO_TILES.get(node, []):
                resource, token = board.tiles[tile_index]
                if resource == "desert":
                    continue
                if token != roll:
                    continue
                if tile_index == board.robber_position:
                    continue  # robber blocks production

                gain = 1 if node in self.settlements else 2
                self.resources[resource] += gain

    def pay_for_build(self, build_type: str):
        """
        Deduct resources from the player for a given build type.
        build_type (str): One of 'settlement', 'city', 'road', or 'dev_card'.
        """
        if build_type not in BUILD_COSTS:
            raise ValueError(f"Unknown build type: {build_type}")

        cost = BUILD_COSTS[build_type]
        # Check affordability first (should already be validated)
        if not self.can_afford(build_type):
            raise ValueError(f"Player {self.color} cannot afford to build {build_type}")

        for resource, amount in cost.items():
            self.resources[resource] -= amount

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