from dataclasses import dataclass

@dataclass
class ActionLogEntry:
    player_name: str
    player_color: str
    text: str
