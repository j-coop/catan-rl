from typing import Dict

from marl.params.catan_constants import RESOURCE_TYPES
from params.catan_constants import DEV_CARD_COUNTS


class CatanBank:
    """
    Tracks remaining resources and development cards.
    """
    def __init__(self):
        self.resources: Dict[str, int] = {res: 19 for res in RESOURCE_TYPES}
        self.dev_cards: Dict[str, int] = DEV_CARD_COUNTS.copy()
