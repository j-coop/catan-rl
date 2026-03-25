import os

from pettingzoo import AECEnv
from gymnasium import spaces
import numpy as np

from marl.env.ActionSpace import ActionSpace
from marl.env.Rewards import Rewards
from params.catan_constants import *
from params.tiles2nodes_adjacency_map import TILES_TO_NODES
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
        self.robber_cf_scale = 2.0
        self.robber_cf_clip = 5.0

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
        tile_token = self.game.board.tiles[tile_index][1]

        if blocks_himself:
            return -5.0

        if tile_token is None:
            return -2.0

        # Check for victims and their yield
        total_victim_yield = 0.0
        has_victim = False
        max_rival_vp = 0
        
        for node_idx in TILES_TO_NODES[tile_index]:
            owner = self.game.board.nodes[node_idx]
            if owner is not None and owner != agent:
                has_victim = True
                owner_player = self.game.get_player(owner)
                multiplier = 2.0 if node_idx in owner_player.cities else 1.0
                total_victim_yield += (DICE_PROBABILITIES[tile_token] / MAX_PROBABILITY) * multiplier
                if owner_player.victory_points > max_rival_vp:
                    max_rival_vp = owner_player.victory_points

        if not has_victim:
            return -2.0
            
        # Base reward from purely blocking production
        # Multiplier of 1.0 ensures average blocks are around +1 to +2 after CF scaling, 
        # and truly amazing blocks hit +3 to +4, keeping them well underneath the -7 self-penalty.
        base_score = (total_victim_yield - 0.4) * 1.0  
        
        # VP bonus: heavily incentivize blocking players close to winning.
        vp_bonus = (max_rival_vp / 10.0) * 0.5 
        
        return base_score + vp_bonus

    def _counterfactual_robber_reward(self, agent: str, chosen_tile_index: int) -> float:
        """
        Calculates the robber reward. We use absolute raw scores
        to ensure 'sensible' moves (blocking strong opponents) are positive
        and others (empty tiles, self-blocking) are negative.
        """
        chosen_raw = self._raw_robber_tile_score(agent, chosen_tile_index)
        
        # Scale the result by the environment's robber_cf_scale
        return self.robber_cf_scale * chosen_raw

    def apply_action(self, agent: str, action: int):
        for spec in self.actions.action_specs:
            start, end = spec.range
            if start <= action < end:
                local_index = action - start
                
                # Use refactored logic for both calculation and execution
                p_before = self.compute_potential(agent)
                context = self._get_action_context(agent, spec.name)
                
                # Execute action handler
                spec.handler(agent, local_index)
                
                p_after = self.compute_potential(agent)
                special_reward = self._calculate_special_reward(agent, spec.name, local_index, context)
                
                # Note: step() will recalculate potential_before/after and final reward.
                # apply_action in AEC usually just mutates. 
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
        settlement_placement_model_path = os.path.abspath(
            os.path.join(
                BASE_DIR,
                "../../../best_models/best_model_init_placement_settlements"
            )
        )
        road_placement_model_path = os.path.abspath(
            os.path.join(
                BASE_DIR,
                "../../../best_models/best_model_init_placement_roads"
            )
        )

        self.game = CatanGame(
            player_colors=[""] * 4,
            player_names=self.agents,
            settlement_placement_model_path=settlement_placement_model_path,
            road_placement_model_path=road_placement_model_path
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
                
                # final_pot = self.compute_potential(a)
                # terminal_reward = self.compute_reward(
                #     a, 
                #     potential_before=final_pot, 
                #     potential_after=0.0, # Irrelevant on term
                #     special_reward=0.0
                # )
                
                # self.rewards[a] = float(terminal_reward)
                # self._cumulative_rewards[a] += terminal_reward

                # Removed terminal Win/Loss rewards for basic training.
                # By not subtracting potential_before, we don't zero out the episode for PBRS.
                # The agent's sum of rewards over the episode will exactly equal its final potential.
                # High potential losers are no longer penalized!
                self.rewards[a] = 0.0
                self._cumulative_rewards[a] += 0.0
        else:
            self.rewards[agent] = float(reward)
            self._cumulative_rewards[agent] += reward

        if self.is_end_turn_action(action):
            self.agent_selection = self.game.current_player.name
            self.game.handle_dice_roll()
