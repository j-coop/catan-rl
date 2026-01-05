from pettingzoo import AECEnv
from gymnasium import spaces
import numpy as np

from marl.env.ActionSpace import ActionSpace
from marl.env.Rewards import Rewards
from params.catan_constants import *
from marl.model.CatanGame import CatanGame
from marl.env.common import EnvActionHandlerMixin


class CatanEnv(AECEnv,
               EnvActionHandlerMixin):
    metadata = {"name": "catan_v1"}

    def __init__(self):
        super().__init__()
        self.colors = [""] * 4
        self.agents = [
            "Blue Player",
            "Purple Player",
            "Yellow Player",
            "Green Player",
        ]
        self.step_counter = 0
        self.possible_agents = self.agents[:]

        self.actions = ActionSpace(self)
        obs_dim = self.get_observation_space_size()
        act_dim = self.actions.get_action_space_size()

        self.observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Box(
                    low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(act_dim,), dtype=np.int8
                ),
            })
            for agent in self.agents
        }

        self.action_spaces = {
            agent: spaces.Discrete(act_dim)
            for agent in self.agents
        }

    def apply_action(self, agent: str, action: int):
        for spec in self.actions.action_specs:
            start, end = spec.range
            if start <= action < end:
                local_index = action - start
                spec.handler(agent, local_index)
                return
        raise ValueError(f"Invalid action index: {action}")

    @staticmethod
    def get_observation_space_size() -> int:
        """Return the flattened size of the observation vector."""
        # Global board (1214) + self (23) + others (42) = 1279
        return 1279

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.agent_selection = self.agents[0]

        self.game = CatanGame(
            player_colors=[""] * 4,
            player_names=self.agents
        )
        self.actions = ActionSpace(self)
        self.reward_object = Rewards(self.game)

        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._cumulative_rewards = {a: 0.0 for a in self.agents}

        self.game.handle_dice_roll()

    def observe(self, agent):
        player = self.game.get_player(agent)

        obs_vec = np.array(self.get_observation(agent), dtype=np.float32)
        mask = np.array(self.actions.get_action_mask(player), dtype=np.int8)

        return {
            "observation": obs_vec,
            "action_mask": mask
        }

    def step(self, action):
        if VERBOSE:
            print("------------------------------------------------------------------------------------")
            print(f"TURN {self.game.turn} - {self.agent_selection}")
            print(f"Resources: {player.resources}")
            print(f"Chosen action: {action}")
        print(f"Step: {self.step_counter}")
        self.step_counter += 1
        agent = self.agent_selection

        self._clear_rewards()

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        player = self.game.get_player(agent)
        mask = self.actions.get_action_mask(player)
        if mask[action] == 0:
            raise ValueError(f"Illegal action {action} for {agent}")

        potential_before = self.compute_potential(agent)
        self.apply_action(agent, action)

        if self.is_end_turn_action(action):
            self.game.end_turn(is_ui_action=False)
            self.agent_selection = self.game.current_player.name
            self.game.handle_dice_roll()

        potential_after = self.compute_potential(agent)
        reward = self.compute_reward(agent, potential_before, potential_after)

        self.rewards[agent] = float(reward)
        self._cumulative_rewards[agent] += reward

        if self.game.game_over:
            for a in self.agents:
                self.terminations[a] = True
