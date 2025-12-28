import random
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np

from marl.model.CatanBoard import CatanBoard
from marl.model.CatanBank import CatanBank
from marl.model.CatanPhase import CatanPhase
from marl.model.CatanPlayer import CatanPlayer
from params.catan_constants import (N_NODES, N_EDGES,
                                    LONGEST_ROAD_MIN_LENGTH, BANK_TRADE_PAIRS,
                                    RESOURCE_TYPES)
from params.edges_list import EDGES_LIST
from params.nodes2tiles_adjacency_map import NODES_TO_TILES


class CatanGame:
    """
    Orchestrates game logic and provides observation/action mapping.
    """
    def __init__(self,
                 player_colors: List[str], 
                 player_names: List[str],
                 ai_players: Optional[List[bool]] = None,
                 training: bool = False):
        self.board: CatanBoard = CatanBoard()
        self.bank: CatanBank = CatanBank()
        self.players = [
            CatanPlayer(name=name, color=color, bank=self.bank)
            for name, color in zip(player_names, player_colors)
        ]
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

        # Special actions control
        self.year_of_plenty_choices_remaining = 0
        self.roads_remaining_from_card = 0

        # For UI
        self.last_roll = None

        print("GAME OBJECT INITIALIZED")

        if not training:
            self.generate_random_init_board_state()
            self.ai_players = [0] * len(self.players)
        else:
            self.ai_players = ai_players

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
        self.last_roll = roll
        print(f"Dice rolled: {roll}")
        if roll == 7:
            self.phase = CatanPhase.ROBBER_MOVE
            for player in self.players:
                player.discard_random_half()
        else:
            for player in self.players:
                player.take_resources(roll, self.board)

    def step(self, player_name: str, action: int):
        # Decode action index, call appropriate place/robber/buy/end turn methods
        # Update game_over if victory condition met
        pass

    def get_legal_actions(self, player: CatanPlayer) -> List[int]:
        # Return list of action indices that are legal
        return []

    def rotate_players(self, current_agent_index: int):
        players = self.players
        rotated = players[current_agent_index:] + players[:current_agent_index]
        return rotated

    def get_observation(self, agent: str) -> Dict[str, np.ndarray]:
        """
        Returns PettingZoo-friendly observation:
        - observation: full numeric features
        - action_mask: valid action indices
        """
        player = self.get_player(agent)
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
        return next((p for p in self.players if p.name == agent), CatanPlayer("", "", self.bank))

    def check_victory(self, agent: str):
        player = self.get_player(agent)
        won = player.victory_points >= 10
        if won:
            self.winner = agent
            self.game_over = True

    def next_turn(self):
        self.turn = (self.turn + 1) % len(self.players)

    def build_settlement(self, agent, node_index, init_placement=False):
        self.board.nodes[node_index] = agent
        player = self.get_player(agent)
        player.settlements.append(node_index)
        if self.board.ports[node_index] is not None:
            player.ports[self.board.ports[node_index]] = True
        if not init_placement:
            player.pay_for_build("settlement")
        player.points += 1
        self.recompute_longest_road()
        self.check_victory(agent)

    def build_city(self, agent, node_index):
        player = self.get_player(agent)
        player.cities.append(node_index)
        player.pay_for_build("city")
        player.points += 1
        self.check_victory(agent)
        player.settlements.remove(node_index)

    def recompute_longest_road(self):
        previous_owner = self.longest_road_owner

        # Reset state
        self.longest_road_owner = None
        self.longest_road_length = 0

        for player in self.players:
            length = self.get_longest_road_length(player.name)
            player.longest_road = length
            if length >= LONGEST_ROAD_MIN_LENGTH and length > self.longest_road_length:
                self.longest_road_length = length
                self.longest_road_owner = player

        # Handle victory points transfer
        if previous_owner != self.longest_road_owner:
            if previous_owner:
                previous_owner.points -= 2
            if self.longest_road_owner:
                self.longest_road_owner.points += 2

    def build_road(self, agent, edge_index, init_placement=False):
        player = self.get_player(agent)
        self.board.edges[edge_index] = agent
        player.roads.append(edge_index)

        if not init_placement:
            if self.roads_remaining_from_card > 0:
                self.roads_remaining_from_card -= 1
                if self.roads_remaining_from_card == 0:
                    self.phase = CatanPhase.NORMAL
            else:
                player.pay_for_build("road")

        self.recompute_longest_road()
        self.check_victory(agent)

    def buy_dev_card(self, agent):
        player = self.get_player(agent)
        card = self.bank.draw_dev_card()
        if card:
            player.dev_cards[card] += 1
            player.pay_for_build("dev_card")

    def play_dev_card(self, agent, card_type):
        player = self.get_player(agent)
        if player.dev_cards[card_type] <= 0:
            raise ValueError(f"{player.name} does not have a {card_type} card to play.")

        player.dev_cards[card_type] -= 1

        # Dispatch to specific handlers
        if card_type == "knight":
            player.knights_played += 1
            # Switch to robber move phase – the player will select where to move next
            self.phase = CatanPhase.ROBBER_MOVE
            # Check for largest army
            if player.knights_played >= 3 and player.knights_played > self.largest_army_count:
                # 2 points for player
                player.points += 2
                self.check_victory(agent)
                if self.largest_army_owner is not None:
                    self.largest_army_owner.points -= 2
                self.largest_army_owner = player
        elif card_type == "road_building":
            # Player will now be able to build two roads
            self.phase = CatanPhase.ROAD_BUILDING
            self.roads_remaining_from_card = 2  # Track roads to build
        elif card_type == "year_of_plenty":
            # Player chooses 2 resources from bank in two consecutive actions
            self.phase = CatanPhase.YEAR_OF_PLENTY
            self.year_of_plenty_choices_remaining = 2
        elif card_type == "monopoly":
            # Player chooses one resource type to take from all others
            self.phase = CatanPhase.MONOPOLY
        elif card_type == "victory_point":
            # Immediate visible point
            player.points += 1
            self.check_victory(agent)
        else:
            raise ValueError(f"Unknown dev card type: {card_type}")

    def play_monopoly(self, player: CatanPlayer, resource: str):
        for other_player in self.players:
            if other_player.name == player.name:
                continue
            stolen = other_player.resources[resource]
            if stolen > 0:
                other_player.resources[resource] = 0
                player.resources[resource] += stolen

        self.phase = CatanPhase.NORMAL

    def give_year_of_plenty_resource(self, player: CatanPlayer, resource: str):
        if self.bank.resources[resource] > 0:
            player.resources[resource] += 1
            self.bank.draw_bank_resource(resource, 1)

        self.year_of_plenty_choices_remaining -= 1
        if self.year_of_plenty_choices_remaining == 0:
            self.phase = CatanPhase.NORMAL

    def move_robber(self, agent_name: str, tile_index: int):
        """
        Move robber to tile_index. Then select a victim among players
        adjacent to that tile using heuristic: choose player with most total resources (if any).
        Steal one random resource from victim (if they have any).
        """
        # Set robber position
        self.board.robber_position = tile_index

        # Going back to normal game flow
        self.phase = CatanPhase.NORMAL

        # Find players adjacent to tile_index (players who have settlement or city on any node touching that tile)
        adjacent_nodes = [node for node, tiles in NODES_TO_TILES.items() if tile_index in tiles]
        adjacent_players = set()
        for node in adjacent_nodes:
            owner = self.board.nodes[node]
            if owner is not None and owner != agent_name:
                adjacent_players.add(owner)

        if not adjacent_players:
            return  # no victims

        # Heuristic: choose player with most total resource cards
        victims = []
        max_resources = -1
        for victim_name in adjacent_players:
            victim_player = self.get_player(victim_name)
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
        thief = self.get_player(agent_name)
        if thief:
            thief.resources[stolen_resource] += 1

    def trade_bank(self, agent, trade_index: int):
        """
        Executes a trade with the bank based on an explicit trade index.
        """
        if not (0 <= trade_index < len(BANK_TRADE_PAIRS)):
            raise ValueError(f"Invalid trade index: {trade_index}")

        give_resource, receive_resource = BANK_TRADE_PAIRS[trade_index]
        self.trade_with_bank(agent, give_resource, receive_resource)

    def trade_with_bank(self, agent, give_resource: str, receive_resource: str):
        """
        Trades resources with the bank using the best available ratio for the player.
        """
        player = self.get_player(agent)
        ratio = player._get_trade_ratio(give_resource)
        if player.resources[give_resource] < ratio:
            raise ValueError(f"{player.name} does not have enough {give_resource} to trade with bank (ratio={ratio}). {player.resources}")

        # Validate bank has the requested resource
        if self.bank.resources[receive_resource] <= 0:
            raise ValueError(f"Bank is out of {receive_resource}! - this should have been masked")

        # Perform trade
        player.resources[give_resource] -= ratio
        player.resources[receive_resource] += 1

        self.bank.resources[give_resource] += ratio
        self.bank.resources[receive_resource] -= 1

    def choose_resource(self, agent, resource_index: int):
        """
        Handles resource selection actions for special dev cards:
        - YEAR_OF_PLENTY: choose two different resources (two consecutive steps)
        - MONOPOLY: choose one resource type to monopolize
        """
        player = self.get_player(agent)
        resource = RESOURCE_TYPES[resource_index]

        # --- YEAR OF PLENTY ---
        if self.phase == CatanPhase.YEAR_OF_PLENTY:
            self.give_year_of_plenty_resource(player, resource)
        # --- MONOPOLY ---
        elif self.phase == CatanPhase.MONOPOLY:
            self.play_monopoly(player, resource)
        else:
            raise RuntimeError("choose_resource called outside a valid special card phase.")

    def take_resource(self, agent, resource: str):
        player = self.get_player(agent)
        player.resources[resource] += 1

    def end_turn(self, agent=None, index=None, is_ui_action=False):
        print(f"Ending turn for {self.current_player.name}")
        self.turn += 1
        if self.turn == 4:
            self.turn = 0

        # If comes from UI - apply turn actions from step
        if is_ui_action:
            self.handle_dice_roll()

    def get_longest_road_length(self, player_name: str) -> int:
        """
        Returns the length of the longest continuous road chain for a player.
        """
        # Build adjacency list of nodes connected by player's roads
        adjacency = defaultdict(list)
        player_edges = [
            (i, e)
            for i, e in enumerate(EDGES_LIST)
            if self.board.edges[i] == player_name
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
                if blocking_owner is not None and blocking_owner != player_name:
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

    def has_player_the_longest_road(self, player):
        return self.longest_road_owner == player

    def has_player_the_largest_army(self, player):
        return self.largest_army_owner == player

    def generate_random_init_board_state(self):

        def place_settlement(player):
            valid_spots = self.board.get_valid_settlement_spots(player, True)
            valid_spot = random.choice(valid_spots)
            self.build_settlement(player.name, valid_spot, True)
            return valid_spot

        def place_road(player, valid_roads):
            valid_road = random.choice(valid_roads)
            self.build_road(player.name, valid_road, True)

        for move_id in range(4):
            player = self.players[move_id]
            node = place_settlement(player)
            valid_roads = self.board.get_valid_road_spots(player)
            place_road(player, valid_roads)

        for move_id in range(4):
            player = self.players[move_id]
            node = place_settlement(player)
            adj_tiles_ids = NODES_TO_TILES.get(node)
            for tile_id in adj_tiles_ids:
                tile = self.board.tiles[tile_id]
                res_name = tile[0]
                res_amount = tile[1]
                if res_amount is not None:
                    player.resources[res_name] += 1
                    self.bank.draw_bank_resource(res_name, 1)

            roads = self.board.get_valid_road_spots(player)
            valid_roads = []
            for road in roads:
                road_nodes = EDGES_LIST[road]
                if road_nodes[0] == node or road_nodes[1] == node:
                    valid_roads.append(road)
            place_road(player, valid_roads)
