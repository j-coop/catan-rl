import random
from typing import List, Dict

import numpy as np

from marl.model.CatanBoard import CatanBoard
from marl.model.CatanBank import CatanBank
from marl.model.CatanPlayer import CatanPlayer
from params.catan_constants import N_NODES, N_EDGES, BUILD_COSTS


class CatanGame:
    """
    Orchestrates game logic and provides observation/action mapping.
    """
    def __init__(self, player_colors: List[str]):
        self.players: List[CatanPlayer] = [CatanPlayer(color) for color in player_colors]
        self.board: CatanBoard = CatanBoard()
        self.bank: CatanBank = CatanBank()
        self.turn: int = 0
        self.game_over: bool = False

    @property
    def current_player(self) -> CatanPlayer:
        return self.players[self.turn]

    def get_dice_roll(self):
        dice_one = random.randint(1, 6)
        dice_two = random.randint(1, 6)
        return dice_one + dice_two

    def handle_dice_roll(self):
        roll = self.get_dice_roll()
        for player in self.players:
            player.take_resources(roll, self.board)

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

    def get_player(self, agent):
        return next((p for p in self.players if p.color == agent), CatanPlayer(""))

    def next_turn(self):
        self.turn = (self.turn + 1) % len(self.players)

    def build_settlement(self, agent, node_index):
        self.board.nodes[node_index] = agent
        player = self.get_player(agent)
        player.settlements.append(node_index)
        player.pay_for_build("settlement")
        player.points += 1

    def build_city(self, agent, node_index):
        player = self.get_player(agent)
        player.cities.append(node_index)
        player.pay_for_build("city")
        player.points += 1
        # Remove settlement
        player.settlements.remove(node_index)

    def build_road(self, agent, edge_index):
        pass

    def buy_dev_card(self, agent):
        pass

    def play_dev_card(self, agent, card_type):
        pass

    def move_robber(self, agent, tile_index):
        pass

    def trade_with_bank(self, agent, give_resource, receive_resource):
        pass

    def end_turn(self, agent):
        pass
