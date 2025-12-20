from typing import List

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from marl.env.ActionSpace import ActionSpace
from marl.env.Rewards import Rewards
from marl.model.CatanPlayer import CatanPlayer
from params.catan_constants import (RESOURCE_TYPES, TILE_TYPES, PORT_TYPES, MAX_RESOURCE_COUNT, MAX_VICTORY_POINTS,
                                    ROADS_PER_PLAYER, SETTLEMENTS_PER_PLAYER, CITIES_PER_PLAYER, MAX_KNIGHTS, GAMMA)
from marl.model.CatanGame import CatanGame


class CatanEnv(MultiAgentEnv):

    metadata = {"name": "catan_v0"}

    def __init__(self, env_config=None):
        super().__init__()
        self.colors = [""] * 4
        self.agents = [
            "Blue Player",
            "Purple Player",
            "Yellow Player",
            "Green Player"
        ]
        self.agent_selection = self.agents[0]
        self.possible_agents = self.agents.copy()

        # Game Logic Layer object
        self.game = CatanGame(player_colors=self.colors,
                              player_names=self.agents)

        self.actions = ActionSpace(self.game)

        self.observation_spaces = {
            agent: spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.get_observation_space_size(),),
                        dtype=np.float32
                    )
            for agent in self.agents
        }

        self.action_spaces = {
            agent: spaces.Discrete(self.actions.get_action_space_size())
            for agent in self.agents
        }

        self.reward_object = Rewards(self.game)

        # First roll is handled manually
        self.pending_dice_roll = False
        self.game.handle_dice_roll()

    # def observation_space(self, agent=None):
    #     print(agent)
    #     if agent is None:
    #         return self.observation_spaces  # returns dict if needed
    #     return self.observation_spaces
    #
    # def action_space(self, agent=None):
    #     if agent is None:
    #         return self.action_spaces
    #     return self.action_spaces

    @property
    def get_sub_environments(self):
        return self.unwrapped

    def seed(self, seed=None):
        """Set the RNG seed for reproducibility."""
        np.random.seed(seed)
        return [seed]

    @staticmethod
    def get_observation_space_size() -> int:
        """Return the flattened size of the observation vector."""
        # Global board (1214) + self (23) + others (42) = 1279
        return 1279

    """
    Handles executing action with given index in action space for given agent
    Calls delegating logic to game object logic layer (CatanGame)
    """
    def apply_action(self, agent: str, action: int):
        for spec in self.actions.action_specs:
            start, end = spec.range
            if start <= action < end:
                print(f"Action type: {spec.name}")
                local_index = action - start
                spec.handler(agent, local_index)
                return
        raise ValueError(f"Invalid action index: {action}")

    """
    One step corresponds to one action (finer control, better action to reward association)
    Only choosing 'end turn' action ends game logic turn
    """
    def step(self, action_dict):

        agent = self.agent_selection
        player = self.game.get_player(agent)
        action = action_dict[agent]
        print("------------------------------------------------------------------------------------")
        print(f"TURN {self.game.turn} - {self.agent_selection}")
        print(f"Resources: {player.resources}")
        print(f"Chosen action: {action}")

        # minimum na teraz - wybiera akcje, nielegalne kończą turę - ponoć nawet stosowane
        mask = self.actions.get_action_mask(player)
        # print(mask)
        if mask[action] == 0:
            print("Chosen action illegal - end turn")
            action = self.actions.get_action_space_size() - 1  # end turn instead of illegal

        potential_before = self.compute_potential(agent)
        self.apply_action(agent, action)

        # Check if this ends the current player's turn
        if self.is_end_turn_action(action):
            # Advance to next player (no dice roll yet)
            self.game.end_turn()
            self.agent_selection = self.game.current_player.name

            # Mark that a dice roll should happen
            self.pending_dice_roll = True
        else:
            # Continue with same agent
            self.agent_selection = agent

        potential_after = self.compute_potential(agent)
        reward = self.compute_reward(agent, potential_before, potential_after)
        self.rewards[agent] = reward

        # IMPORTANT for PettingZoo bookkeeping:
        # self._accumulate_rewards()
        self._cumulative_rewards[agent] += self.rewards[agent]

        # Handle dice roll if necessary
        if self.pending_dice_roll:
            self.game.handle_dice_roll()
            self.pending_dice_roll = False

        # Observations
        obs = {self.agent_selection: self.observe(self.agent_selection)}

        # RLlib expects a dict of rewards for all agents
        rewards = {p.name: self.rewards.get(p.name, 0.0) for p in self.game.players}

        # RLlib expects terminateds, truncateds dict with "__all__"
        terminateds = {p.name: self.terminations.get(p.name, False) for p in self.game.players}
        terminateds["__all__"] = all(terminateds.values())

        truncateds = {p.name: self.truncations.get(p.name, False) for p in self.game.players}
        truncateds["__all__"] = all(truncateds.values())

        # Info dict per agent
        infos = {self.agent_selection: self.infos.get(self.agent_selection, {})}

        return obs, rewards, terminateds, truncateds, infos

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.game = CatanGame(player_colors=self.colors,
                              player_names=self.agents)

        self.agent_selection = self.agents[0]
        self._agent_iterator = iter(self.agents)

        # REQUIRED for PettingZoo
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Handle initial dice roll
        # We want the first player to begin after a dice roll occurs.
        self.game.handle_dice_roll()

        # Generate the initial observation for the first agent
        # obs = {
        #     agent: self._build_agent_observation(self.game.get_player(agent))
        #     for agent in self.agents
        # }
        # infos = {k: {} for k in obs.keys()}
        # return obs, infos
        # return (
        #     {self.agent_selection: self.observe(self.agent_selection)},
        #     {},
        # )
        obs = {self.agent_selection: self.observe(self.agent_selection)}
        return obs, {}

    def observe(self, agent):
        """
        Return the observation dict for `agent`.
        We return:
            {
                "observation": np.array([...], dtype=np.float32),
                "action_mask": np.array([...], dtype=np.int8)
            }
        This function expects CatanGame to provide:
         - game.get_observation(agent) -> numpy array
        """
        # observation vector
        obs_vec = np.array(self.get_observation(agent), dtype=np.float32)

        # player = self.game.get_player(agent)
        # mask = np.array(self.actions.get_action_mask(player), dtype=np.int8)
        # print(f"MASK: {mask}")

        # return {
        #     "observation": obs_vec,
        #     "action_mask": mask
        # }
        return obs_vec

    def render(self):
        pass

    def state(self):
        pass

    def close(self):
        pass

    # =========== ACTION HANDLERS ===========
    # Delegating action handling to CatanGame
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
        # For simplicity, we can map 0–4 to known dev card types.
        card_types = ["knight", "road_building", "year_of_plenty", "monopoly", "victory_point"]
        card_type = card_types[card_index]
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
        resource = RESOURCE_TYPES[resource_index]
        self.game.take_resource(agent, resource)

    def end_turn(self):
        """End the player's turn."""
        self.game.end_turn()

    def is_end_turn_action(self, action):
        return action == self.actions.get_action_space_size() - 1

    def compute_potential(self, agent):
        return self.reward_object.compute_potential(agent)

    def compute_reward(self, agent, potential_before, potential_after, gamma=GAMMA) -> float:
        if self.game.game_over and self.game.winner == agent:
            # Return max out of scale reward for actual win
            return 1000.0
        else:
            return (gamma * potential_after) - potential_before

    def get_observation(self, agent: str) -> np.ndarray:
        """Encodes full game state into a flat vector for the given agent."""
        player_index = self.agents.index(agent)
        players = self.game.rotate_players(player_index)

        global_features = self.encode_global_board()
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

    def encode_global_board(self) -> np.ndarray:
        board = self.game.board
        num_players = len(self.agents)

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
                owner_onehot[self.agents.index(edge)] = 1.0
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
                owner_onehot[self.agents.index(node)] = 1.0
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
        dev_counts = np.array(list(player.dev_cards.values()), dtype=np.float32) / 5.0

        victory_points = np.array([player.victory_points / MAX_VICTORY_POINTS])
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
                    len(p.dev_cards) / 10.0,
                    p.hidden_points / MAX_VICTORY_POINTS,
                    float(has_longest_road),
                    float(has_largest_army),
                    p.knights_played / MAX_KNIGHTS
                ], dtype=np.float32),
                port_flags
            ])
            features.append(feats)
        return np.concatenate(features)
