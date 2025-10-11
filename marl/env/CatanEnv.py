import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from marl.model.CatanGame import CatanGame
from marl.params.catan_constants import N_NODES, N_EDGES


class CatanEnv(AECEnv):

    metadata = {"name": "catan_v0"}

    def __init__(self):
        super().__init__()
        self.agents = ['blue', 'red', 'white', 'black']
        self.possible_agents = self.agents[:]
        self.agent_selection = self.agents[0]

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

    @staticmethod
    def get_action_space_size() -> int:
        size = 0
        # N_NODES for placing settlements and cities each
        size += 2 * N_NODES
        # N_EDGES for placing roads
        size += N_EDGES
        # 1 for buy dev card
        # 5 for playing dev cards
        # 1 for move robber (steal included)
        # 1 for end turn
        size += 1 + 5 + 1 + 1
        return size

    @staticmethod
    def get_observation_space_size() -> int:
        size = 0
        # TODO: define observation space shape
        return size


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
