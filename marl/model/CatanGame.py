from typing import List, Dict

import numpy as np

from marl.model import CatanBoard
from marl.model.CatanBank import CatanBank
from marl.model.CatanPlayer import CatanPlayer
from marl.params.catan_constants import N_NODES, N_EDGES


class CatanGame:
    """
    Orchestrates game logic and provides observation/action mapping.
    """
    def __init__(self, player_names: List[str]):
        self.players: List[CatanPlayer] = [CatanPlayer(name) for name in player_names]
        self.board: CatanBoard = CatanBoard()
        self.bank: CatanBank = CatanBank()
        self.turn: int = 0
        self.current_player: CatanPlayer = self.players[self.turn]
        self.game_over: bool = False

    def step(self, player_name: str, action: int):
        # Decode action index, call appropriate place/robber/buy/end turn methods
        # Update game_over if victory condition met
        pass

    def get_legal_actions(self, player: CatanPlayer) -> List[int]:
        # Return list of action indices that are legal
        return []

    def get_observation(self, player: CatanPlayer) -> Dict[str, np.ndarray]:
        """
        Returns PettingZoo-friendly observation:
        - observation: full numeric features
        - action_mask: valid action indices
        """
        obs = np.concatenate([
            self.board.get_board_observation(),
            player.get_observation()
        ])
        legal_actions = self.get_legal_actions(player)
        action_mask = np.zeros(self.get_action_space_size(), dtype=np.int8)
        action_mask[legal_actions] = 1
        return {"observation": obs, "action_mask": action_mask}

    @staticmethod
    def get_action_space_size() -> int:
        size = 2 * N_NODES + N_EDGES + 1 + 5 + 1 + 1
        return size

    def next_turn(self):
        self.turn = (self.turn + 1) % len(self.players)
        self.current_player = self.players[self.turn]
