from dataclasses import dataclass
from typing import Callable, Tuple

@dataclass
class ActionSpec:
    name: str
    range: Tuple[int, int]  # (start_index, end_index)
    handler: Callable       # method to call when executed
