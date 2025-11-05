import os
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.base_env.env import CatanBaseEnv
from envs.init_placement_env.env import CatanInitPlacementEnv


# -------------------------------
#  Mask function for masked PPO
# -------------------------------
def mask_fn(env) -> np.ndarray:
    return env.get_action_masks()


# -------------------------------
#  Helper: Run evaluation loop
# -------------------------------
def evaluate_model(model_path: str, base_env_obs) -> float:
    # Load model
    model = MaskablePPO.load(model_path)

    # Create placement env and wrap with ActionMasker
    placement_env = CatanInitPlacementEnv(base_env_obs=base_env_obs, train=False)
    placement_env = ActionMasker(placement_env, mask_fn)

    obs, _ = placement_env.reset()
    total_reward = 0.0
    model_name = os.path.basename(model_path).replace(".zip", "")

    # Run 16 placement steps (8 settlement-road pairs)
    for _ in range(16):
        mask = placement_env.unwrapped.get_action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)

        obs, reward, done, truncated, info = placement_env.step(action)
        total_reward += reward

        if done or truncated:
            break
    return total_reward


# -------------------------------
#  Main execution
# -------------------------------
if __name__ == "__main__":

    model_paths = [
        "trained_models/init-placement/ppo_mask_20251103_163113.zip",
        "trained_models/init-placement/ppo_mask_20251104_131634.zip",
        "trained_models/init-placement/ppo_mask_20251104_004624.zip",
        "trained_models/init-placement/ppo_mask_20251104_223029.zip",
        "trained_models/init-placement/ppo_mask_20251105_045845.zip",
        "trained_models/init-placement/ppo_mask_20251105_180903.zip",
    ]
    rewards = [0] * len(model_paths)

    number_of_runs = 80
    for _ in range(number_of_runs):
        base_env = CatanBaseEnv(save_env=True)
        base_env_obs = base_env.reset()

        for i in range(len(model_paths)):
            total_reward = evaluate_model(model_paths[i], base_env_obs)
            rewards[i] += total_reward

    # Summary
    print("\n=== Evaluation Summary ===")
    for i in range(len(model_paths)):
        print(f"{model_paths[i]}: avg reward = {(rewards[i] / number_of_runs):.3f}")
