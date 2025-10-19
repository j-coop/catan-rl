import random
from typing import Dict, List

from params.catan_constants import (DEV_CARD_COUNTS,
                                    RESOURCE_TYPES)


class CatanBank:
    """
    Tracks remaining resources and development cards.
    """
    def __init__(self, seed: int | None = None):
        self.resources: Dict[str, int] = {res: 19 for res in RESOURCE_TYPES}

        # Initialize dev card stack
        self.dev_cards_stack: List[str] = []
        for card, count in DEV_CARD_COUNTS.items():
            self.dev_cards_stack.extend([card] * count)

        rng = random.Random(seed)
        rng.shuffle(self.dev_cards_stack)

    def draw_dev_card(self) -> str | None:
        """
        Draw and return the next dev card from the shuffled stack.
        Returns None if stack is empty.
        """
        if not self.dev_cards_stack:
            return None
        return self.dev_cards_stack.pop()

    def remaining_dev_cards(self) -> int:
        """
        Returns number of remaining dev cards.
        """
        return len(self.dev_cards_stack)
