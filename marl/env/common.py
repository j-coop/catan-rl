import numpy as np
from typing import Dict, List

from marl.model.CatanPlayer import CatanPlayer
from params.catan_constants import *


class EnvActionHandlerMixin:

    def build_settlement(self, agent: str, node_index: int):
        """Place a new settlement on the board at the specified node index."""
        self.game.build_settlement(agent, node_index)

    def build_city(self, agent: str, node_index: int):
        """Upgrade an existing settlement owned by the player into a city."""
        self.game.build_city(agent, node_index)

    def build_road(self, agent: str, edge_index: int):
        """Build a road on the specified edge if connected to the player’s network."""
        self.game.build_road(agent, edge_index)

    def buy_dev_card(self, agent: str, _: int):
        """Purchase a development card if resources and stock allow."""
        self.game.buy_dev_card(agent)

    def play_dev_card(self, agent: str, card_index: int):
        """Play a development card of the specified type."""
        card_type = DEV_CARD_TYPES[card_index]
        self.game.play_dev_card(agent, card_type)

    def move_robber(self, agent: str, tile_index: int):
        """Move the robber to a specified tile and perform the stealing logic."""
        self.game.move_robber(agent, tile_index)

    def trade_bank(self, agent: str, trade_index: int):
        """Trade with the bank (or ports if owned) using the best available ratio."""
        # There are 20 possible trades = 5 give × 4 receive (excluding same resource).
        # Map flat index → (give_resource, receive_resource)
        give_idx = trade_index // 4
        recv_idx = trade_index % 4
        give_resource = RESOURCE_TYPES[give_idx]
        receive_resource = [r for r in RESOURCE_TYPES if r != give_resource][recv_idx]
        self.game.trade_with_bank(agent, give_resource, receive_resource)

    def choose_resource(self, agent: str, resource_index: int):
        if resource_index >= len(RESOURCE_TYPES):
            raise ValueError("No such resource")
        self.game.choose_resource(agent, resource_index)

    def end_turn(self, _: str, __: int, is_ui_action=False):
        """End the player's turn."""
        self.game.end_turn(is_ui_action=is_ui_action)

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
            robber_flag = 1.0 if board.robber_position == i or tile_token == 7 or tile_token is None else 0.0
            tile_feats.append(np.concatenate([res_onehot, [number_val, robber_flag]]))
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

        port_flags = np.zeros(len(PORT_TYPES), dtype=np.float32)
        owned_nodes = list(player.settlements) + list(player.cities)
        for node_idx in owned_nodes:
            port_type = self.game.board.ports[node_idx]
            if port_type in PORT_TYPES:
                port_flags[PORT_TYPES.index(port_type)] = 1.0

        return np.concatenate([
            res_counts,
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
