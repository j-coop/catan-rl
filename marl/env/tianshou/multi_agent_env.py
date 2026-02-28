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
        self.robber_cf_scale = 0.35
        self.robber_cf_clip = 1.5

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

    def _raw_robber_tile_score(self, agent: str, tile_index: int) -> float:
        players_around_tile = self.game.board.get_players_around_tile(tile_index)
        blocks_himself = agent in players_around_tile
        num_players_around = len(players_around_tile)
        tile_token = self.game.board.tiles[tile_index][1]

        if blocks_himself:
            return -1.0
        if num_players_around == 0 or tile_token is None:
            return -0.5

        num_players_weight = 0.7 + num_players_around * 0.1
        token_weight = DICE_PROBABILITIES[tile_token] / MAX_PROBABILITY
        return 0.5 + 0.5 * (num_players_weight * token_weight)

    def _counterfactual_robber_reward(self, agent: str, chosen_tile_index: int) -> float:
        move_robber_spec = next(spec for spec in self.actions.action_specs if spec.name == "move_robber")
        start, end = move_robber_spec.range

        player = self.game.get_player(agent)
        mask = self.actions.get_action_mask(player)
        legal_tile_indices = [idx - start for idx in range(start, end) if mask[idx]]
        if not legal_tile_indices:
            legal_tile_indices = list(range(end - start))

        raw_scores = np.array(
            [self._raw_robber_tile_score(agent, tile_idx) for tile_idx in legal_tile_indices],
            dtype=np.float32
        )
        chosen_raw = self._raw_robber_tile_score(agent, chosen_tile_index)
        centered = chosen_raw - float(np.mean(raw_scores))
        normalized = centered / (float(np.std(raw_scores)) + 1e-6)
        clipped = float(np.clip(normalized, -self.robber_cf_clip, self.robber_cf_clip))
        return self.robber_cf_scale * clipped

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
                        special_reward = -3.0
                elif spec.name == "build_settlement":
                    special_reward = 4.0
                elif spec.name == "build_city":
                    special_reward = 3.0
                elif spec.name == "build_road":
                    special_reward = 1.5
                elif spec.name == "end_turn":
                    special_reward = 0.0
                    # if self.game.current_player
                    # cards_num = self.game.current_player.total_cards
                    # if cards_num > 7:
                    #     special_reward -= 0.2 * (cards_num - 7)
                elif spec.name == "play_dev_card":
                    if local_index in [2, 3, 4]:
                        special_reward = 0.5
                elif spec.name == "choose_resource":
                    resource = RESOURCE_TYPES[local_index]
                    if resource not in self.game.current_player.produced_resources:
                        special_reward = 0.4
                    else:
                        special_reward = 0.0
                elif spec.name == "move_robber":
                    special_reward = self._counterfactual_robber_reward(agent, local_index)
                spec.handler(agent, local_index)
                return special_reward
        raise ValueError(f"Invalid action index: {action}")

    @staticmethod
    def get_observation_space_size() -> int:
        """Return the flattened size of the observation vector."""
        return FULL_ACTION_SPACE_SIZE

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

        potential_after = self.compute_potential(agent)
        reward = self.compute_reward(agent,
                                     potential_before,
                                     potential_after,
                                     special_reward=special_reward)
        if VERBOSE:
            print(f"Potential before: {potential_before}, after: {potential_after},"
                  f" diff: {potential_after - potential_before}, reward: {reward}")

        if self.game.game_over:
            for a in self.agents:
                self.terminations[a] = True
                
                # Fetch potentials to cleanly zero-out the episode as per PBRS terminal state math
                final_pot = self.compute_potential(a)
                terminal_reward = self.compute_reward(
                    a, 
                    potential_before=final_pot, 
                    potential_after=0.0, # Irrelevant on term
                    special_reward=0.0
                )
                
                self.rewards[a] = float(terminal_reward)
                self._cumulative_rewards[a] += terminal_reward
        else:
            self.rewards[agent] = float(reward)
            self._cumulative_rewards[agent] += reward

        if self.is_end_turn_action(action):
            self.agent_selection = self.game.current_player.name
            self.game.handle_dice_roll()
