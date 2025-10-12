import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from params.catan_constants import N_NODES, N_EDGES
from marl.model.CatanGame import CatanGame
from marl.util.ActionSpec import ActionSpec


class CatanEnv(AECEnv):

    metadata = {"name": "catan_v0"}

    def __init__(self):
        super().__init__()
        self.agents = ['blue', 'red', 'white', 'black']
        self.possible_agents = self.agents[:]
        self.agent_selection = self.agents[0]

        # Mapping of action space to action handlers
        self.action_specs: list[ActionSpec] = []
        # Action specs initialization
        self.action_space_size = 0
        self.init_action_specs()

        # Game Logic Layer object
        self.game = CatanGame(self.agents)

        self.action_spaces = {
            agent: spaces.Discrete(self.get_action_space_size()) for agent in self.agents
        }

        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(self.action_spaces[agent].n,), dtype=np.int8
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

    def init_action_specs(self):
        start = 0

        self.action_specs.append(ActionSpec("build_settlement", (start, start + N_NODES), self.build_settlement))
        start += N_NODES

        self.action_specs.append(ActionSpec("build_city", (start, start + N_NODES), self.build_city))
        start += N_NODES

        self.action_specs.append(ActionSpec("build_road", (start, start + N_EDGES), self.build_road))
        start += N_EDGES

        self.action_specs.append(ActionSpec("buy_dev_card", (start, start + 1), self.buy_dev_card))
        start += 1

        self.action_specs.append(ActionSpec("play_dev_card", (start, start + 5), self.play_dev_card))
        start += 5

        self.action_specs.append(ActionSpec("move_robber", (start, start + 19), self.move_robber))
        start += 19

        self.action_specs.append(ActionSpec("trade_bank", (start, start + 20), self.trade_bank))
        start += 20

        self.action_specs.append(ActionSpec("end_turn", (start, start + 1), self.end_turn))
        start += 1

        self.action_space_size = start

    @staticmethod
    def get_action_space_size() -> int:
        size = 0
        # N_NODES for placing settlements and cities each
        size += 2 * N_NODES
        # N_EDGES for placing roads
        size += N_EDGES
        # 1 for buy dev card
        # 5 for playing dev cards
        # 19 for moving robber to each field (steal included)
        # 20 for trading with bank (each resource for each resource)
        # 1 for end turn
        size += 1 + 5 + 19 + 20 + 1
        return size

    @staticmethod
    def get_observation_space_size() -> int:
        size = 0
        # TODO: define observation space shape
        return size

    """
    Handles executing action with given index in action space for given agent
    Calls delegating logic to game object logic layer (CatanGame)
    """
    def apply_action(self, agent: str, action: int):
        for spec in self.action_specs:
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
            self.game.end_turn()
            self.agent_selection = self.game.current_player.color

            # Mark that a dice roll should happen
            self.pending_dice_roll = True
        else:
            # Continue with same agent
            self.agent_selection = agent

        # Compute rewards
        reward = self.compute_reward(agent)
        self.rewards[agent] = reward

        # Determine next agent
        self.agent_selection = self._next_agent()

        # Handle dice roll if necessary
        if self.pending_dice_roll:
            self.game.handle_dice_roll()
            self.pending_dice_roll = False

        # Generate observation for next agent (includes state after player's dice roll)
        obs = self.observe(self.agent_selection)

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
        resource_types = ["wood", "brick", "sheep", "wheat", "ore"]
        give_idx = trade_index // 4
        recv_idx = trade_index % 4
        give_resource = resource_types[give_idx]
        receive_resource = [r for r in resource_types if r != give_resource][recv_idx]
        self.game.trade_with_bank(agent, give_resource, receive_resource)

    def end_turn(self, agent: str, _: int):
        """End the player's turn."""
        self.game.end_turn(agent)
