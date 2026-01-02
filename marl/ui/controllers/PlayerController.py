from abc import ABC, abstractmethod

from marl.model.GameManager import GameManager


class PlayerController(ABC):
    def __init__(self, player_name: str):
        self.player_name = player_name

    @abstractmethod
    def request_action(self, game, action_space, game_manager: GameManager):
        """
        Called when it's this player's turn.
        Must eventually trigger a game action.
        """
        pass
