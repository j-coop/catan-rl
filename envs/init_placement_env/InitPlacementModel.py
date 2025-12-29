from typing import List

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.base_env.env import CatanBaseEnv
from envs.init_placement_env.env import CatanInitPlacementEnv
from marl.adapters.game_to_base_env import game_to_base_env_state
from marl.model.CatanBoard import CatanBoard
from params.catan_constants import N_NODES
from params.edges_list import EDGES_LIST
from params.tiles2nodes_adjacency_map import TILES_TO_NODES


def mask_fn(env) -> np.ndarray:
    return env.get_action_masks()

"""
Init placement model adapter class
"""
class InitPlacementModel:
    def __init__(self, model_path: str, board: CatanBoard):
        self.model = MaskablePPO.load(model_path)
        self.board = board

        # Order of placing settlements and roads
        self.settlements_history = []
        self.roads_history = []

    def generate_initial_board(self):
        """
        Returns base_env_obs AFTER all placements
        """
        # Base state from CatanGame
        base_state = game_to_base_env_state(self.board)

        # Create base env
        base_env = CatanBaseEnv(save_env=False, initial_state=base_state)
        base_obs = base_env.reset()

        # Create init placement env
        placement_env = CatanInitPlacementEnv(
            base_env_obs=base_obs,
            train=False,
        )
        placement_env = ActionMasker(placement_env, mask_fn)

        obs, _ = placement_env.reset()

        # Run placement loop
        for i in range(16):
            mask = placement_env.unwrapped.get_action_masks()
            action, _ = self.model.predict(
                obs,
                deterministic=True,
                action_masks=mask,
            )
            player_index = placement_env.unwrapped.turn_order[i // 2]
            if i % 2 == 0:
                # Settlement
                self.settlements_history.append((int(action), player_index)) # (node_id, player_index)
            else:
                # Road
                self.roads_history.append((int(action) - N_NODES, player_index)) # (edge_id, player_index)
            obs, _, _, _, info = placement_env.step(action)

        # base_obs inside env is now mutated
        return placement_env.unwrapped._base_obs

    def apply_base_obs_to_game(self, base_obs, game, player_order=None):
        """
        Apply final base_obs from init placement env to CatanGame.

        base_obs["nodes_owners"]: shape (N_TILES, 6, N_PLAYERS)
        base_obs["edges_owners"]: shape (N_TILES, 6, N_PLAYERS)
        player_order: optional list mapping base_obs player index → CatanGame player name
        """

        return self.settlements_history, self.roads_history
