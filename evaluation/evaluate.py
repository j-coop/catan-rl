import os
import time
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.init_placement_env.road_wrapper import CatanRoadPlacementEnv
from envs.init_placement_env.env import CatanInitPlacementEnv
from visualization.map_plotter import CatanMapPlotter


# Action mask function
def mask_fn(env) -> np.ndarray:
    return env.get_action_masks()

# Load trained model
model = MaskablePPO.load("C:\\PG\\sem_8\\PB\\trained_models\\init-placement\\ppo_mask_20251123_185652")

env = CatanInitPlacementEnv()
env.reset()
env = ActionMasker(CatanRoadPlacementEnv(env), mask_fn)

# Reset environment (unpack Gym-style tuple)
obs, _ = env.reset()

# Create timestamped output directory
timestamp = time.strftime("%Y%m%d-%H%M%S")
save_dir = f"placement_runs/{timestamp}"
os.makedirs(save_dir, exist_ok=True)


for placement_step in range(8):
    # Get current valid action mask
    mask = env.unwrapped.get_action_masks()

    # Predict with action mask
    action, _states = model.predict(obs, deterministic=True, action_masks=mask)
    print(f"Step {placement_step}: Chosen action {action}")

    # Step environment
    obs, reward, done, truncated, info = env.step(action)

    # Save map image every pair (after road)
    filename = os.path.join(save_dir, f"step_{placement_step:02d}.png")
    plotter = CatanMapPlotter(info['base_obs'])
    plotter.plot_catan_map(filename)
