from typing import Dict

from marl.params.catan_constants import RESOURCE_TYPES, DEV_CARD_TYPES


class CatanBank:
    """
    Tracks remaining resources and development cards.
    """
    def __init__(self):
        self.resources: Dict[str, int] = {res: 19 for res in RESOURCE_TYPES}
        self.dev_cards: Dict[str, int] = {card: 2 for card in DEV_CARD_TYPES}  # simplified
