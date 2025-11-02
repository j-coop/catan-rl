import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from marl.env.ActionSpace import ActionSpace
from params.catan_constants import (RESOURCE_TYPES)
from marl.model.CatanGame import CatanGame


class CatanEnv(AECEnv):

    metadata = {"name": "catan_v0"}

    def __init__(self):
        super().__init__()
        self.agents = ['blue', 'red', 'white', 'black']
        self.possible_agents = self.agents[:]
        self.agent_selection = self.agents[0]

        # Game Logic Layer object
        self.game = CatanGame(self.agents)

        # Action space handling object
        self.actions = ActionSpace(self.game)

        self.action_spaces = {
            agent: spaces.Discrete(self.actions.get_action_space_size()) for agent in self.agents
        }

        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(self.action_spaces[agent].shape[0],), dtype=np.int8
                    ),
                    "observation": spaces.Box(
                        low=0, high=1, shape=(self.get_observation_space_size(),), dtype=np.float32
                    ),
                }
            )
            for agent in self.agents
        }

        # First roll is handled manually
        self.pending_dice_roll = False
        self.game.handle_dice_roll()

    @staticmethod
    def get_observation_space_size() -> int:
        size = 0
        # TODO: define observation space shape
        return size

    def get_spec_for_action(self, agent: str, action: int):
        for spec in self.actions.action_specs:
            start, end = spec.range
            if start <= action < end:
                local_index = action - start
                # spec.handler(agent, local_index)
                return spec
        raise ValueError(f"Invalid action index: {action}")

    """
    Handles executing action with given index in action space for given agent
    Calls delegating logic to game object logic layer (CatanGame)
    """
    def apply_action(self, agent: str, action: int):
        for spec in self.actions.action_specs:
            start, end = spec.range
            if start <= action < end:
                local_index = action - start
                spec.handler(agent, local_index)
                return
        raise ValueError(f"Invalid action index: {action}")

    """
    One step corresponds to one action (finer control, better action to reward association)
    Only choosing 'end turn' action ends game logic turn
    """
    def step(self, action):
        agent = self.agent_selection
        player = self.game.current_player

        # Apply the action in the game logic
        self.apply_action(agent, action)

        # Check if this ends the current player's turn
        if self.is_end_turn_action(action):
            # Advance to next player (no dice roll yet)
            self.game.end_turn(agent)
            self.agent_selection = self.game.current_player.color

            # Mark that a dice roll should happen
            self.pending_dice_roll = True
        else:
            # Continue with same agent
            self.agent_selection = agent

        # Compute rewards
        reward = self.compute_reward(agent)
        self.rewards[agent] = reward

        # Handle dice roll if necessary
        if self.pending_dice_roll:
            self.game.handle_dice_roll()
            self.pending_dice_roll = False

        # Generate observation for next agent (includes state after player's dice roll)
        obs = self.observe(self.agent_selection)

        # Masking actions depending on game state and phase - dynamically updated inside ActionSpace
        mask = self.actions.get_action_mask()
        obs["action_mask"] = mask

        return obs, reward, self.terminations[agent], self.truncations[agent], {}

    def reset(self, seed=None, options=None):
        # Reset PettingZoo base state
        super().reset(seed=seed)

        # Reset game logic layer
        self.game = CatanGame(self.agents)

        self.agent_selection = self.agents[0]

        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Handle initial dice roll
        # We want the first player to begin after a dice roll occurs.
        self.game.handle_dice_roll()

        # Generate the initial observation for the first agent
        obs = self.observe(self.agent_selection)
        return obs

    def observe(self, agent):
        pass

    def render(self):
        pass

    def state(self):
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

    def end_turn(self, agent: str, _: int):
        """End the player's turn."""
        self.game.end_turn(agent)

    def is_end_turn_action(self, action):
        return action == self.actions.get_action_space_size() - 1

    def compute_reward(self, agent) -> int:
        return 0
