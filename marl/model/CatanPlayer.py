from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np
import random

from params.nodes2tiles_adjacency_map import NODES_TO_TILES
from params.catan_constants import (DEV_CARD_TYPES, PORT_TYPES,
                                    RESOURCE_TYPES, BUILD_COSTS)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from marl.model.CatanBoard import CatanBoard
    from marl.model.CatanBank import CatanBank


class CatanPlayer:
    """
    Stores player-specific game state.
    """
    def __init__(self, name: str, color: str, bank: CatanBank):
        self.name: str = name
        self.color: str = color
        self.bank: CatanBank = bank
        self.resources: Dict[str, int] = {res: 0 for res in RESOURCE_TYPES}

        # Dev cards
        self.dev_cards: Dict[str, int] = {card: 0 for card in DEV_CARD_TYPES}
        self.knights_played: int = 0

        self.ports: Dict[str, bool] = {port: False for port in PORT_TYPES}
        self.settlements: List[int] = []  # node indices
        self.cities: List[int] = []       # node indices
        self.roads: List[int] = []        # edge indices
        self.points: int = 0
        self.hidden_points: int = 0 # victory points other players don't know about
        self.longest_road = 0

    @property
    def victory_points(self) -> int:
        return self.points + self.hidden_points

    def discard_random_half(self):
        total = sum(self.resources.values())
        if total <= 7:
            return

        to_discard = total // 2
        pool = []
        for res, count in self.resources.items():
            pool.extend([res] * count)

        # Randomly discard
        discarded = random.sample(pool, to_discard)
        for res in discarded:
            self.resources[res] -= 1
            self.bank.return_bank_resource(res, 1)

    def take_resources(self, roll: int, board: CatanBoard):
        """
        Collect resources for this player after a dice roll.
        Settlements yield +1, cities yield +2.
        Resources are only given if the bank has enough.
        """
        for node in self.settlements + self.cities:
            for tile_index in NODES_TO_TILES.get(node, []):
                resource, token = board.tiles[tile_index]
                if resource == "desert":
                    continue
                if token != roll:
                    continue
                if tile_index == board.robber_position:
                    continue

                gain = 1 if node in self.settlements else 2
                bank_available = self.bank.resources[resource]
                actual_gain = min(gain, bank_available)

                self.resources[resource] += actual_gain
                self.bank.draw_bank_resource(resource, actual_gain)

                if actual_gain < gain:
                    print(f"{self.name} gets {actual_gain}/{gain} {resource} "
                        f"(bank now {self.bank.resources[resource]})")

    def pay_for_build(self, build_type: str):
        print(f"Available resources: {self.resources}")
        if build_type not in BUILD_COSTS:
            raise ValueError(f"Unknown build type: {build_type}")

        cost = Counter(BUILD_COSTS[build_type])
        shortages = self._deduct_direct_resources(cost)

        if not self._all_shortages_covered(shortages):
            shortages = self._cover_shortages_with_trades(shortages)

        if not self._all_shortages_covered(shortages):
            raise ValueError(f"Player {self.name} cannot afford to build {build_type}, even with trades, {self.resources}")

    def _deduct_direct_resources(self, cost: Counter) -> Counter:
        """Deduct resources the player already has for a given cost.
        Returns a Counter with remaining resources still needed."""
        remaining = Counter()
        for res, qty in cost.items():
            available_qty = self.resources.get(res, 0)
            if available_qty >= qty:
                self.resources[res] -= qty
                self.bank.return_bank_resource(res, qty)
                remaining[res] = 0
            else:
                self.resources[res] = 0
                remaining[res] = qty - available_qty
                self.bank.return_bank_resource(res, available_qty)

        return remaining

    def _all_shortages_covered(self, shortages: Counter) -> bool:
        """Check if all values in shortages are 0"""
        return all(v == 0 for v in shortages.values())

    def _cover_shortages_with_trades(self, shortages: Counter) -> Counter:
        """Perform optimal trades to cover shortages using only bank/ports.
        Returns the updated shortages counter."""
        tradable_resources = []

        for res, qty in self.resources.items():
            if qty == 0:
                continue
            ratio = self._get_trade_ratio(res)
            if qty >= ratio:
                tradable_resources.append((ratio, res, qty))

        # Best trade ratios first (2:1, then 3:1, then 4:1)
        tradable_resources.sort(key=lambda x: x[0])

        for ratio, res, qty in tradable_resources:
            max_trade_units = qty // ratio
            if max_trade_units == 0:
                continue

            for shortage_res, needed in shortages.items():
                if needed == 0:
                    continue

                trade_units = min(max_trade_units, needed)

                self.resources[res] -= trade_units * ratio
                self.bank.return_bank_resource(res, trade_units * ratio)

                shortages[shortage_res] -= trade_units
                max_trade_units -= trade_units

                if max_trade_units == 0:
                    break

        return shortages

    def _get_trade_ratio(self, resource: str) -> int:
        """Return the best trade ratio available for a given resource."""
        if self.ports.get(resource, False):
            return 2
        elif self.ports.get("3for1", False):
            return 3
        else:
            return 4

    def can_afford_directly(self, build_type: str) -> bool:
        """
        Check if the player can afford a given build (settlement, city, road, dev_card),
        considering only posessed resources
        """
        cost = Counter(BUILD_COSTS[build_type])
        available = Counter(self.resources)
        if all(available[res] >= qty for res, qty in cost.items()):
            return True
        else:
            return False

    def can_afford_with_trades(self, build_type: str, bank: CatanBank) -> bool:
        """
        Check if the player can afford a given build (settlement, city, road, dev_card),
        considering posessed resources and trades with bank
        """
        # print(f"can_afford_with_trades: {build_type}, resources: {self.resources}")
        cost = Counter(BUILD_COSTS[build_type])
        available = Counter(self.resources)
        shortages = {res: max(0, qty - available[res]) for res, qty in cost.items()}
        total_needed = sum(shortages.values())

        #  The bank must have enough resources to give the missing ones
        for res, missing in shortages.items():
            if missing > 0 and bank.resources.get(res, 0) < missing:
                return False

        tradable = 0
        for res, qty in available.items():
            required = cost.get(res, 0)
            surplus = max(0, qty - required)

            if surplus == 0:
                continue

            ratio = self._get_trade_ratio(res)
            tradable += surplus // ratio

        # print("Is enough resources", build_type, tradable >= total_needed, self.resources)
        return tradable >= total_needed

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

    def get_playable_dev_cards(self):
        return [card_type for card_type in DEV_CARD_TYPES 
                if self.dev_cards.get(card_type, 0) > 0]

    def get_valid_bank_trades(self) -> List[(str, str)]:
        """
        Returns all valid (give_resource, receive_resource) trade pairs
        the player can perform with the bank or their ports.
        """
        valid_trades: List[(str, str)] = []

        for give in RESOURCE_TYPES:
            amount = self.resources.get(give, 0)

            # Skip if no resources of that type
            if amount <= 1:
                continue

            # Determine trade ratio for this resource
            if self.ports.get(give, False):
                ratio = 2
            elif self.ports.get("3for1", False):
                ratio = 3
            else:
                ratio = 4

            # Check if player can afford trade
            if amount < ratio:
                continue

            # Can trade only when bank have enough resources
            for receive in RESOURCE_TYPES:
                if receive == give:
                    continue
                if self.bank.resources.get(receive, 0) == 0:
                    continue
                valid_trades.append((give, receive))

        return valid_trades
