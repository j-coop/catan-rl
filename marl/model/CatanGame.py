import random
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np

from marl.model.CatanBoard import CatanBoard
from marl.model.CatanBank import CatanBank
from marl.model.CatanPhase import CatanPhase
from marl.model.CatanPlayer import CatanPlayer
from params.catan_constants import N_NODES, N_EDGES, LONGEST_ROAD_MIN_LENGTH
from params.edges_list import EDGES_LIST
from params.nodes2tiles_adjacency_map import NODES_TO_TILES


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
        self.winner: str | None = None

        self.phase = CatanPhase.NORMAL
        self.phase_actor: Optional[CatanPlayer] = None
        self.phase_data: dict = {}

        # Longest road
        self.longest_road_length = 0
        self.longest_road_owner: CatanPlayer | None = None

        # Largest army
        self.largest_army_count = 0
        self.largest_army_owner: CatanPlayer | None = None

    @property
    def current_player(self) -> CatanPlayer:
        return self.players[self.turn]

    @staticmethod
    def get_dice_roll():
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
        size = 2 * N_NODES + N_EDGES + 1 + 5 + 19 + 20 + 5 + 1
        return size

    def get_player(self, agent):
        return next((p for p in self.players if p.color == agent), CatanPlayer(""))

    def check_victory(self, agent: str):
        player = self.get_player(agent)
        won = player.victory_points >= 10
        if won:
            self.winner = agent
            self.game_over = True

    def next_turn(self):
        self.turn = (self.turn + 1) % len(self.players)

    def build_settlement(self, agent, node_index):
        self.board.nodes[node_index] = agent
        player = self.get_player(agent)
        player.settlements.append(node_index)
        player.pay_for_build("settlement")
        player.points += 1
        self.check_victory(agent)

    def build_city(self, agent, node_index):
        player = self.get_player(agent)
        player.cities.append(node_index)
        player.pay_for_build("city")
        player.points += 1
        self.check_victory(agent)
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
                self.check_victory(agent)
                previous_holder = self.longest_road_owner
                previous_holder.points -= 2
            self.longest_road_owner = player

    def buy_dev_card(self, agent):
        player = self.get_player(agent)
        card = self.bank.draw_dev_card()
        if card:
            player.dev_cards[card] += 1
            player.pay_for_build("dev_card")

    def play_dev_card(self, agent, card_type):
        player = self.get_player(agent)
        if player.dev_cards[card_type] <= 0:
            raise ValueError(f"{player.color} does not have a {card_type} card to play.")

        player.dev_cards[card_type] -= 1

        # Dispatch to specific handlers
        if card_type == "knight":
            player.knights_played += 1
            self.move_robber(agent, 0) # TODO: swap with real action flow - masking to 19 move robber actions
            # Check for largest army
            if player.knights_played >= 3 and player.knights_played > self.largest_army_count:
                # 2 points for player
                player.points += 2
                self.check_victory(agent)
                previous_holder = self.largest_army_owner
                previous_holder.points -= 2
                self.largest_army_owner = player
        elif card_type == "road_building":
            # TODO: next two steps: build a road
            self.handle_road_building_card(player)
        elif card_type == "year_of_plenty":
            # TODO: next two steps: choose resource to get from bank
            self.handle_year_of_plenty_card(player)
        elif card_type == "monopoly":
            # TODO: next step: choose one resource to demand from others
            self.handle_monopoly_card(player)
        elif card_type == "victory_point":
            player.hidden_points += 1
            self.check_victory(agent)
        else:
            raise ValueError(f"Unknown dev card type: {card_type}")

    def move_robber(self, agent_color: str, tile_index: int):
        """
        Move robber to tile_index. Then select a victim among players
        adjacent to that tile using heuristic: choose player with most total resources (if any).
        Steal one random resource from victim (if they have any).
        """
        # Set robber position
        self.board.robber_position = tile_index

        # Find players adjacent to tile_index (players who have settlement or city on any node touching that tile)
        adjacent_nodes = [node for node, tiles in NODES_TO_TILES.items() if tile_index in tiles]
        adjacent_players = set()
        for node in adjacent_nodes:
            owner = self.board.nodes[node]
            if owner is not None and owner != agent_color:
                adjacent_players.add(owner)

        if not adjacent_players:
            return  # no victims

        # Heuristic: choose player with most total resource cards
        victims = []
        max_resources = -1
        for victim_color in adjacent_players:
            victim_player = self.get_player(victim_color)
            if victim_player is None:
                continue
            total = sum(victim_player.resources.values())
            if total > max_resources:
                victims = [victim_player]
                max_resources = total
            elif total == max_resources:
                victims.append(victim_player)

        # If there's at least one victim with resources, steal one random resource from one selected victim
        if not victims or max_resources == 0:
            return

        victim = random.choice(victims)
        # choose a random resource the victim has (>0)
        available = [r for r, cnt in victim.resources.items() if cnt > 0]
        if not available:
            return

        stolen_resource = random.choice(available)
        victim.resources[stolen_resource] -= 1
        thief = self.get_player(agent_color)
        if thief:
            thief.resources[stolen_resource] += 1

    def trade_with_bank(self, agent, give_resource: str, receive_resource: str):
        """
        Trades resources with the bank using the best available ratio for the player.
        """
        player = self.get_player(agent)

        # Determine trade ratio (lowest available)
        if player.ports.get(give_resource, False):
            ratio = 2  # 2:1 specific port
        elif player.ports.get("3for1", False):
            ratio = 3  # 3:1 general port
        else:
            ratio = 4  # 4:1 default bank rate

        # Validate resource availability
        if player.resources[give_resource] < ratio:
            raise ValueError(f"{player.color} does not have enough {give_resource} to trade with bank.")

        # Validate bank has the requested resource
        if self.bank.resources[receive_resource] <= 0:
            raise ValueError(f"Bank is out of {receive_resource}! - this should have been masked")

        # Perform trade
        player.resources[give_resource] -= ratio
        player.resources[receive_resource] += 1

        self.bank.resources[give_resource] += ratio
        self.bank.resources[receive_resource] -= 1

    def take_resource(self, agent, resource: str):
        player = self.get_player(agent)
        player.resources[resource] += 1

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
