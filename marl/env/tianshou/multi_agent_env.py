import os

from pettingzoo import AECEnv
from gymnasium import spaces
import numpy as np

from marl.env.ActionSpace import ActionSpace
from marl.env.Rewards import Rewards
from params.catan_constants import *
from marl.model.CatanGame import CatanGame
from marl.env.common import EnvActionHandlerMixin

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

        self.shaping_weight = SHAPING_WEIGHT_START
        self._shaping_step = 0
        self._shaping_anneal_steps = SHAPING_ANNEAL_STEPS
        self.shaping_weight_end = SHAPING_WEIGHT_END
        self.win_reward = WIN_REWARD
        self.loss_penalty = LOSS_PENALTY

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
        special_reward = 0
        for spec in self.actions.action_specs:
            start, end = spec.range
            if start <= action < end:
                local_index = action - start
                if VERBOSE:
                    print(f"Action type: {spec.name} - {local_index}")
                if spec.name == "trade_bank":
                    give, take = BANK_TRADE_PAIRS[local_index]
                    if self.game.current_player.is_bad_trade(give, take):
                        special_reward = -2.0
                elif spec.name == "build_settlement":
                    special_reward = 1.0
                elif spec.name == "build_city":
                    special_reward = 1.0
                elif spec.name == "build_road":
                    special_reward = 1.5
                elif spec.name == "play_dev_card":
                    special_reward = 2.0
                elif spec.name == "end_turn":
                    special_reward = 0.0
                    cards_num = self.game.current_player.total_cards
                    if cards_num > 7:
                        special_reward -= 0.4 * (cards_num - 7)
                elif spec.name == "move_robber":
                    players_around_tile = self.game.board.get_players_around_tile(local_index)
                    blocks_himself = agent in players_around_tile
                    num_players_around = len(players_around_tile)
                    tile_token = self.game.board.tiles[local_index][1]
                    if blocks_himself:
                        special_reward = -1.0  # penalty for blocking yourself
                    elif num_players_around == 0 or tile_token is None:
                        special_reward = -0.5  # penalty for not hurting anyone (i know it sounds bad)
                    else:
                        num_players_weight = 0.7 + num_players_around * 0.1
                        token_weight = DICE_PROBABILITIES[tile_token] / MAX_PROBABILITY
                        special_reward = 0.5 + 0.5 * (num_players_weight * token_weight)
                spec.handler(agent, local_index)
                return special_reward
        raise ValueError(f"Invalid action index: {action}")

    @staticmethod
    def get_observation_space_size() -> int:
        """Return the flattened size of the observation vector."""
        # Global board (1214) + self (23) + others (42) = 1279
        return 1279

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.agent_selection = self.agents[0]

        init_placement_model_path = os.path.abspath(
            os.path.join(
                BASE_DIR,
                "../../../model/trained_models/init-placement/init_placement_model"
            )
        )

        self.game = CatanGame(
            player_colors=[""] * 4,
            player_names=self.agents,
            init_placement_model_path=init_placement_model_path
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

    def _update_shaping_weight(self):
        if self._shaping_anneal_steps <= 0:
            return
        progress = min(self._shaping_step / self._shaping_anneal_steps, 1.0)
        self.shaping_weight = (
            SHAPING_WEIGHT_START
            + (self.shaping_weight_end - SHAPING_WEIGHT_START) * progress
        )
        self._shaping_step += 1

    def step(self, action):
        agent = self.agent_selection
        player = self.game.get_player(agent)
        if VERBOSE:
            print("------------------------------------------------------------------------------------")
            print(f"TURN {self.game.turn} - {self.agent_selection}")
            print(f"Resources: {player.resources}")
            print(f"Chosen action: {action}")
            print(f"Step: {self.step_counter}")
        self.step_counter += 1

        self._clear_rewards()
        self._update_shaping_weight()

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        mask = self.actions.get_action_mask(player)
        if mask[action] == 0:
            raise ValueError(f"Illegal action {action} for {agent}")

        potential_before = self.compute_potential(agent)
        special_reward = self.apply_action(agent, action)

        if self.is_end_turn_action(action):
            self.agent_selection = self.game.current_player.name
            self.game.handle_dice_roll()

        potential_after = self.compute_potential(agent)
        reward = self.compute_reward(agent, potential_before, potential_after, special_reward=special_reward)
        if VERBOSE:
            print(f"Potential before: {potential_before}, after: {potential_after}, "
                  f"diff: {potential_after - potential_before}, reward: {reward}")

        if self.game.game_over:
            for a in self.agents:
                self.terminations[a] = True
                terminal_reward = self.win_reward if a == self.game.winner else self.loss_penalty
                self.rewards[a] = float(terminal_reward)
                self._cumulative_rewards[a] += terminal_reward
        else:
            self.rewards[agent] = float(reward)
            self._cumulative_rewards[agent] += reward
