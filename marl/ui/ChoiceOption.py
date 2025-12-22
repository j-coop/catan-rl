from dataclasses import dataclass
from typing import Callable


@dataclass
class ChoiceOption:
    text: str
    callback: Callable[[], None]
    enabled: bool = True
