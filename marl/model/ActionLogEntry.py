from dataclasses import dataclass

@dataclass
class ActionLogEntry:
    player_index: int
    player_name: str
    player_color: str
    text: str
