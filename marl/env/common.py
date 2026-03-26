import numpy as np
from typing import Dict, List, Callable, Set, TYPE_CHECKING
if TYPE_CHECKING:
    from marl.model.CatanGame import CatanGame
    from marl.env.Rewards import Rewards

from marl.model.CatanPlayer import CatanPlayer
from params.catan_constants import (
    TILE_TYPES, DICE_PROBABILITIES, PORT_TYPES, DEV_CARD_TYPES,
    DEV_CARD_COUNTS, MAX_RESOURCE_COUNT, MAX_VICTORY_POINTS,
    ROADS_PER_PLAYER, SETTLEMENTS_PER_PLAYER, CITIES_PER_PLAYER,
    MAX_KNIGHTS, RESOURCE_TYPES, BANK_TRADE_PAIRS, VERBOSE, 
    GAMMA, MAX_PROBABILITY
)
from params.tiles2nodes_adjacency_map import TILES_TO_NODES


class EnvActionHandlerMixin:
    game: 'CatanGame'
    reward_object: 'Rewards'
    agents: List[str]
    actions: 'ActionSpace'

    def get_observation_space_size(self) -> int:
        raise NotImplementedError()

    def _get_action_context(self, agent: str, action_name: str) -> Dict:
        """Capture state before action if needed for rewards (e.g. road building)."""
        context = {}
        if action_name == "build_road":
            player = self.game.get_player(agent)
            context["spots_before"] = set(self.game.board.get_valid_settlement_spots(player))
        return context

    def _calculate_special_reward(self, agent: str, action_name: str, local_index: int, context: Dict) -> float:
        """Calculate the 'special_reward' (extra heuristic) for an action."""
        special_reward = 0.0
        player = self.game.get_player(agent)
        bank = self.game.bank

        # Replicate logic from apply_action
        if action_name == "trade_bank":
            give, take = BANK_TRADE_PAIRS[local_index]
            if take not in player.produced_resources:
                # Good trade for missing resource
                special_reward = 2.0 if player.total_cards >= 7 else 1.0
            elif player.is_bad_trade(give, take):
                special_reward = -3.0
        elif action_name == "build_settlement":
            base = 0.5
            prod_values = self.reward_object.production_at_node(local_index).values()
            quality = sum(prod_values)
            special_reward = base + (quality * 20.0)
            
            port_type = self.game.board.ports[local_index]
            if port_type is not None:
                special_reward += 0.5
                if port_type == "3for1":
                    special_reward += 2.5
                elif port_type in RESOURCE_TYPES:
                    player_prod = self.reward_object.production_at_node(local_index).get(port_type, 0.0)
                    for node in player.settlements:
                        player_prod += self.reward_object.production_at_node(node).get(port_type, 0.0)
                    for node in player.cities:
                        player_prod += 2.0 * self.reward_object.production_at_node(node).get(port_type, 0.0)
                    special_reward += player_prod * 25.0
        elif action_name == "build_city":
            base = 0.5
            prod_values = self.reward_object.production_at_node(local_index).values()
            quality = sum(prod_values)
            special_reward = base + (quality * 14.0)
        elif action_name == "build_road":
            special_reward = 1.0
            # Context-based road reward
            spots_after = set(self.game.board.get_valid_settlement_spots(player))
            new_spots = context.get("spots_before", set())
            added_spots = spots_after - new_spots
            if added_spots:
                quality = sum([sum(self.reward_object.production_at_node(s).values()) for s in added_spots])
                special_reward += 2.5 + quality * 3.0
        elif action_name == "end_turn":
            special_reward = 0.0
        elif action_name == "play_dev_card":
            if local_index in [2, 3, 4]:
                special_reward = 2.5
            else:
                special_reward = 1.5
        elif action_name == "choose_resource":
            resource = RESOURCE_TYPES[local_index]
            if resource not in player.produced_resources:
                special_reward = 0.4
            else:
                special_reward = 0.0
        elif action_name == "move_robber":
            if hasattr(self, "_counterfactual_robber_reward"):
                special_reward = self._counterfactual_robber_reward(agent, local_index)

        # Opportunity cost penalties (not checking for settlements when possible)
        if action_name not in ("build_settlement", "build_city", "choose_resource", "move_robber"):
            spots_avail = set(self.game.board.get_valid_settlement_spots(player))
            can_afford_set = player.can_afford_directly("settlement") or player.can_afford_with_trades("settlement", bank)
            can_afford_city = player.can_afford_directly("city") or player.can_afford_with_trades("city", bank)
            
            can_build_set = can_afford_set and player.settlements_remaining > 0 and len(spots_avail) > 0
            can_build_city = can_afford_city and player.cities_remaining > 0 and len(player.settlements) > 0
            
            if can_build_set or can_build_city:
                special_reward += -5.0

        return special_reward

    def _execute_with_reward_log(self, agent: str, action_name: str, local_index: int, mutation_fn: Callable):
        """Wrapper to call mutation and log the resulting RL reward."""
        if not hasattr(self, 'reward_object') or self.reward_object is None:
            mutation_fn()
            return

        p_before = self.compute_potential(agent)
        context = self._get_action_context(agent, action_name)
        
        mutation_fn()
        
        p_after = self.compute_potential(agent)
        special_reward = self._calculate_special_reward(agent, action_name, local_index, context)
        reward = self.compute_reward(agent, p_before, p_after, special_reward=special_reward)
        
        if VERBOSE:
             print(f"\n[RL REWARD] Action: {action_name}({local_index})")
             print(f"  Potential: {p_before:.4f} -> {p_after:.4f} (diff: {p_after - p_before:.4f})")
             print(f"  Special: {special_reward:.4f}")
             print(f"  FINAL REWARD (PBRS): {reward:.4f}")
             print("-" * 30)

    def build_settlement(self, agent: str, node_index: int):
        """Place a new settlement on the board at the specified node index."""
        self._execute_with_reward_log(agent, "build_settlement", node_index,
                                    lambda: self.game.build_settlement(agent, node_index))

    def build_city(self, agent: str, node_index: int):
        """Upgrade an existing settlement owned by the player into a city."""
        self._execute_with_reward_log(agent, "build_city", node_index,
                                    lambda: self.game.build_city(agent, node_index))

    def build_road(self, agent: str, edge_index: int):
        """Build a road on the specified edge if connected to the player’s network."""
        self._execute_with_reward_log(agent, "build_road", edge_index,
                                    lambda: self.game.build_road(agent, edge_index))

    def buy_dev_card(self, agent: str, _: int = 0):
        """Purchase a development card if resources and stock allow."""
        self._execute_with_reward_log(agent, "buy_dev_card", _,
                                    lambda: self.game.buy_dev_card(agent))

    def play_dev_card(self, agent: str, card_index: int):
        """Play a development card of the specified type."""
        card_type = DEV_CARD_TYPES[card_index]
        self._execute_with_reward_log(agent, "play_dev_card", card_index,
                                    lambda: self.game.play_dev_card(agent, card_type))

    def move_robber(self, agent: str, tile_index: int):
        """Move the robber to a specified tile and perform the stealing logic."""
        self._execute_with_reward_log(agent, "move_robber", tile_index,
                                    lambda: self.game.move_robber(agent, tile_index))

    def trade_bank(self, agent: str, trade_index: int):
        """Trade with the bank (or ports if owned) using the best available ratio."""
        # There are 20 possible trades = 5 give × 4 receive (excluding same resource).
        # Map flat index → (give_resource, receive_resource)
        give_idx = trade_index // 4
        recv_idx = trade_index % 4
        give_resource = RESOURCE_TYPES[give_idx]
        receive_resource = [r for r in RESOURCE_TYPES if r != give_resource][recv_idx]
        self._execute_with_reward_log(agent, "trade_bank", trade_index,
                                    lambda: self.game.trade_with_bank(agent, give_resource, receive_resource))

    def choose_resource(self, agent: str, resource_index: int):
        if resource_index >= len(RESOURCE_TYPES):
            raise ValueError("No such resource")
        self._execute_with_reward_log(agent, "choose_resource", resource_index,
                                    lambda: self.game.choose_resource(agent, resource_index))

    def end_turn(self, agent: str, __: int = 0, is_ui_action=False):
        """End the player's turn."""
        self._execute_with_reward_log(agent, "end_turn", __,
                                    lambda: self.game.end_turn(is_ui_action=is_ui_action))

    def is_end_turn_action(self, action):
        return action == self.actions.get_action_space_size() - 1

    def compute_potential(self, agent):
        return self.reward_object.compute_potential(agent)

    def compute_reward(self, agent, potential_before, potential_after, gamma=GAMMA, special_reward=None) -> float:
        # PBRS rule: R = Direct_Reward + (Gamma * PotentialAfter) - PotentialBefore
        shaped_reward = (gamma * potential_after) - potential_before

        # Additive heuristic loop holes
        direct_reward = special_reward if special_reward is not None else 0.0

        total_reward = direct_reward + shaped_reward

        # Clipping reward to given range
        total_reward = 7.0 if total_reward > 7.0 else total_reward
        total_reward = -7.0 if total_reward < -7.0 else total_reward

        return total_reward

    def get_observation(self, agent: str) -> np.ndarray:
        """Encodes full game state into a flat vector for the given agent."""
        player_index = self.agents.index(agent)
        rotated_agent_names = self.agents[player_index:] + self.agents[:player_index]
        players = self.game.rotate_players(player_index)

        global_features = self.encode_global_board(rotated_agent_names)
        self_features = self.encode_self_info(players[0])
        others_features = self.encode_others_info(players[1:])

        obs = np.concatenate([
            global_features,
            self_features,
            others_features
        ])
        assert obs.shape[0] == self.get_observation_space_size(), \
            f"Unexpected observation size: {obs.shape[0]}"
        return obs.astype(np.float32)

    def _build_relative_owner_index(self, rotated_agent_names: List[str]) -> Dict[str, int]:
        return {name: i for i, name in enumerate(rotated_agent_names)}

    def encode_global_board(self, rotated_agent_names: List[str]) -> np.ndarray:
        board = self.game.board
        num_players = len(rotated_agent_names)
        owner_to_relative_idx = self._build_relative_owner_index(rotated_agent_names)

        # --- Tiles ---
        tile_feats = []
        for i, tile in enumerate(board.tiles):
            tile_res = tile[0]
            res_onehot = np.zeros(len(TILE_TYPES))
            if tile_res in TILE_TYPES:
                res_onehot[TILE_TYPES.index(tile_res)] = 1.0
            # normalize number_token to [0,1]
            tile_token = tile[1]
            number_val = tile_token / 12.0 if tile_token is not None else 0
            robber_flag = 1.0 if board.robber_position == i else 0.0

            # Spatial awareness: direct production info on this tile
            self_prod = 0.0
            opp_prod = 0.0
            for node_idx in TILES_TO_NODES[i]:
                owner_name = board.nodes[node_idx]
                if owner_name is None:
                    continue
                
                # yield at this node
                node_token = board.tiles[i][1]
                prob = DICE_PROBABILITIES.get(node_token, 0)
                
                # Check building type via player object
                owner_player = self.game.get_player(owner_name)
                multiplier = 2.0 if node_idx in owner_player.cities else 1.0
                
                prod_yield = (prob / MAX_PROBABILITY) * multiplier
                if owner_name == rotated_agent_names[0]:
                    self_prod += prod_yield
                else:
                    opp_prod += prod_yield
                
            # Binary flags for "presence" (easier for net to parse than small floats)
            self_has_building = 1.0 if self_prod > 0 else 0.0
            opp_has_building = 1.0 if opp_prod > 0 else 0.0

            tile_feats.append(np.concatenate([
                res_onehot, 
                [number_val, robber_flag, self_prod, opp_prod, self_has_building, opp_has_building]
            ]))
        tile_feats = np.concatenate(tile_feats)

        # --- Roads ---
        road_feats = []
        for edge in board.edges:
            owner_onehot = np.zeros(num_players + 1)
            if edge is not None:
                owner_onehot[owner_to_relative_idx[edge]] = 1.0
            else:
                owner_onehot[-1] = 1.0
            road_feats.append(owner_onehot)
        road_feats = np.concatenate(road_feats)

        # --- Nodes ---
        node_feats = []
        for i, node in enumerate(board.nodes):
            # owner encoding
            owner_onehot = np.zeros(num_players + 1)
            if node is not None:
                owner_onehot[owner_to_relative_idx[node]] = 1.0
            else:
                owner_onehot[-1] = 1.0

            # building type two-hot
            building = np.zeros(2)
            building_owner = self.game.get_player(node)
            if building_owner.settlements.count(i) > 0:
                building[0] = 1.0
            elif building_owner.cities.count(i) > 0:
                building[1] = 1.0

            # port one-hot (6 flags)
            port_onehot = np.zeros(len(PORT_TYPES))
            if board.ports[i] in PORT_TYPES:
                port_onehot[PORT_TYPES.index(board.ports[i])] = 1.0
            node_feats.append(np.concatenate([owner_onehot, building, port_onehot]))

        node_feats = np.concatenate(node_feats)

        return np.concatenate([tile_feats, road_feats, node_feats])

    def encode_self_info(self, player: CatanPlayer) -> np.ndarray:
        res_counts = np.array(list(player.resources.values()), dtype=np.float32) / MAX_RESOURCE_COUNT
        dev_max = np.array([DEV_CARD_COUNTS[card] for card in DEV_CARD_TYPES], dtype=np.float32)
        dev_counts = np.array([player.dev_cards[card] for card in DEV_CARD_TYPES]) / dev_max
        victory_points = np.array(
            [min(player.victory_points / MAX_VICTORY_POINTS, 1.0)],
            dtype=np.float32
        )
        has_longest_road = self.game.longest_road_owner is not None and self.game.longest_road_owner.name == player.name
        has_largest_army = self.game.largest_army_owner is not None and self.game.largest_army_owner.name == player.name
        longest_road = np.array([float(has_longest_road)])
        largest_army = np.array([float(has_largest_army)])

        built_structs = np.array([
            len(player.roads) / ROADS_PER_PLAYER,
            len(player.settlements) / SETTLEMENTS_PER_PLAYER,
            len(player.cities) / CITIES_PER_PLAYER
        ])

        knights_played = np.array([player.knights_played / MAX_KNIGHTS])
        total_resources = np.array([player.total_cards / 20.0], dtype=np.float32)

        # Production probabilities (capped at 1.0)
        prod_probs = player.get_production_probs(self.game.board)
        prod_feats = np.array([min(prod_probs[res], 1.0) for res in RESOURCE_TYPES], dtype=np.float32)

        port_flags = np.zeros(len(PORT_TYPES), dtype=np.float32)
        owned_nodes = list(player.settlements) + list(player.cities)
        for node_idx in owned_nodes:
            port_type = self.game.board.ports[node_idx]
            if port_type in PORT_TYPES:
                port_flags[PORT_TYPES.index(port_type)] = 1.0

        return np.concatenate([
            total_resources,
            res_counts,
            prod_feats,
            dev_counts,
            victory_points,
            longest_road,
            largest_army,
            built_structs,
            knights_played,
            port_flags
        ])

    def encode_others_info(self, others: List[CatanPlayer]) -> np.ndarray:
        features = []

        for p in others:
            has_longest_road = self.game.longest_road_owner is not None and self.game.longest_road_owner.name == p.name
            has_largest_army = self.game.largest_army_owner is not None and self.game.largest_army_owner.name == p.name

            port_flags = np.zeros(len(PORT_TYPES), dtype=np.float32)
            owned_nodes = list(p.settlements) + list(p.cities)
            for node_idx in owned_nodes:
                port_type = self.game.board.ports[node_idx]
                if port_type in PORT_TYPES:
                    port_flags[PORT_TYPES.index(port_type)] = 1.0

            feats = np.concatenate([
                np.array([
                    len(p.roads) / ROADS_PER_PLAYER,
                    len(p.settlements) / SETTLEMENTS_PER_PLAYER,
                    len(p.cities) / CITIES_PER_PLAYER,
                    sum(p.dev_cards.values()) / 10.0,
                    p.points / MAX_VICTORY_POINTS,
                    float(has_longest_road),
                    float(has_largest_army),
                    p.knights_played / MAX_KNIGHTS
                ], dtype=np.float32),
                port_flags
            ])
            features.append(feats)
        return np.concatenate(features)
