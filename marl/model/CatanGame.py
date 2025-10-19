import random
from collections import defaultdict
from typing import List, Dict

import numpy as np

from marl.model.CatanBoard import CatanBoard
from marl.model.CatanBank import CatanBank
from marl.model.CatanPlayer import CatanPlayer
from params.catan_constants import N_NODES, N_EDGES, BUILD_COSTS, LONGEST_ROAD_MIN_LENGTH
from params.edges_list import EDGES_LIST


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

        # Longest road
        self.longest_road_length = 0
        self.longest_road_owner: CatanPlayer | None = None

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
        player = self.get_player(agent)
        self.board.edges[edge_index] = agent
        player.roads.append(edge_index)
        player.pay_for_build("road")
        # Check for longest road
        player_longest_road = self.get_longest_road_length(agent)
        if player_longest_road > self.longest_road_length:
            self.longest_road_length = player_longest_road
            if player_longest_road >= LONGEST_ROAD_MIN_LENGTH:
                # 2 points for player
                player.points += 2
                previous_holder = self.longest_road_owner
                previous_holder.points -= 2
            self.longest_road_owner = player

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

    def get_longest_road_length(self, player_color: str) -> int:
        """
        Returns the length of the longest continuous road chain for a player.
        """
        # Build adjacency list of nodes connected by player's roads
        adjacency = defaultdict(list)
        player_edges = [
            (i, e)
            for i, e in enumerate(EDGES_LIST)
            if self.board.edges[i] == player_color
        ]

        if not player_edges:
            return 0

        for i, (a, b) in player_edges:
            adjacency[a].append((b, i))
            adjacency[b].append((a, i))

        # Helper DFS to explore the longest chain
        def dfs(node: int, visited_edges: set) -> int:
            max_len = 0
            for neighbor, edge_idx in adjacency[node]:
                if edge_idx in visited_edges:
                    continue

                # Check if another player's settlement blocks path
                blocking_owner = self.board.nodes[neighbor]
                if blocking_owner is not None and blocking_owner != player_color:
                    continue

                new_visited = visited_edges | {edge_idx}
                length = 1 + dfs(neighbor, new_visited)
                max_len = max(max_len, length)

            return max_len

        # Try DFS from every node that belongs to the player's road network
        max_chain = 0
        for node in adjacency:
            max_chain = max(max_chain, dfs(node, set()))

        return max_chain
