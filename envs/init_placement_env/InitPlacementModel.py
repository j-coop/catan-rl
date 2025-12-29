import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.base_env.env import CatanBaseEnv
from envs.init_placement_env.env import CatanInitPlacementEnv


def mask_fn(env) -> np.ndarray:
    return env.get_action_masks()

"""
Init placement model adapter class
"""
class InitPlacementModel:
    def __init__(self, model_path: str):
        self.model = MaskablePPO.load(model_path)

    def generate_initial_board(self):
        """
        Returns base_env_obs AFTER all placements
        """
        # 1. Create base env
        base_env = CatanBaseEnv(save_env=False)
        base_obs = base_env.reset()

        # 2. Create placement env
        placement_env = CatanInitPlacementEnv(
            base_env_obs=base_obs,
            train=False,
        )
        placement_env = ActionMasker(placement_env, mask_fn)

        # 3. Reset
        obs, _ = placement_env.reset()

        # 4. Run placement loop
        for _ in range(16):
            mask = placement_env.unwrapped.get_action_masks()
            action, _ = self.model.predict(
                obs,
                deterministic=True,
                action_masks=mask,
            )
            obs, _, _, _, info = placement_env.step(action)

        # base_obs inside env is now mutated
        return placement_env.unwrapped.base_obs
