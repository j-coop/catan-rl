import os
import time
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.base_env.env import CatanBaseEnv
from envs.init_placement_env.env import CatanInitPlacementEnv


# Action mask function
def mask_fn(env) -> np.ndarray:
    return env.get_action_masks()

# Load trained model
model = MaskablePPO.load("trained_models/init-placement/init-model.zip")

# Create random base env
base_env = CatanBaseEnv(save_env=True)
base_env_obs = base_env.reset()

# Create placement env and wrap with ActionMasker
placement_env = CatanInitPlacementEnv(base_env_obs=base_env_obs)
placement_env = ActionMasker(placement_env, mask_fn)

# Reset environment (unpack Gym-style tuple)
obs, _ = placement_env.reset()

# Create timestamped output directory
timestamp = time.strftime("%Y%m%d-%H%M%S")
save_dir = f"placement_runs/{timestamp}"
os.makedirs(save_dir, exist_ok=True)

# Run 16-step placement (8 settlement-road pairs)
for placement_step in range(16):
    # Get current valid action mask
    mask = placement_env.unwrapped.get_action_masks()

    # Predict with action mask
    action, _states = model.predict(obs, deterministic=True, action_masks=mask)
    print(f"Step {placement_step}: Chosen action {action}")

    # Step environment
    obs, reward, done, truncated, info = placement_env.step(action)

    # Save map image every pair (after road)
    if placement_step % 2 == 1:
        filename = os.path.join(save_dir, f"step_{placement_step:02d}.png")
        # plot_map(placement_env.unwrapped, filename)
        print(f"Saved map after step {placement_step} to {filename}")
