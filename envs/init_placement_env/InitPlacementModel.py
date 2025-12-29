import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.base_env.env import CatanBaseEnv
from envs.init_placement_env.env import CatanInitPlacementEnv
from marl.adapters.game_to_base_env import game_to_base_env_state
from marl.model.CatanBoard import CatanBoard


def mask_fn(env) -> np.ndarray:
    return env.get_action_masks()

"""
Init placement model adapter class
"""
class InitPlacementModel:
    def __init__(self, model_path: str, board: CatanBoard):
        self.model = MaskablePPO.load(model_path)
        self.board = board

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
        for _ in range(16):
            mask = placement_env.unwrapped.get_action_masks()
            print(mask)
            action, _ = self.model.predict(
                obs,
                deterministic=True,
                action_masks=mask,
            )
            obs, _, _, _, info = placement_env.step(action)

        # base_obs inside env is now mutated
        return placement_env.unwrapped._base_obs

    @staticmethod
    def apply_base_obs_to_game(base_obs, game):
        """
        Takes final base_obs from init placement env and applies it to CatanGame
        """
        print(base_obs["nodes_settlements"])
        print(base_obs["edges_owners"])

        for player_name, node_idx in base_obs["nodes_settlements"]:
            game.build_settlement(player_name, node_idx, init_placement=True)

        for player_name, edge_idx in base_obs["edges_owners"]:
            game.build_road(player_name, edge_idx, init_placement=True)

