import gymnasium as gym
import numpy as np
from datetime import datetime
import torch

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO, MultiInputPolicy

from envs.base_env.env import CatanBaseEnv
from envs.init_placement_env.env import CatanInitPlacementEnv
from params.catan_constants import N_EPISODES, STEPS_PER_EPISODE


def mask_fn(_env: gym.Env) -> np.ndarray:
    return _env.get_action_masks()

base_env = CatanBaseEnv(save_env=True)
base_env_obs = base_env.reset()

# Create custom env
env = CatanInitPlacementEnv()
env.reset()

# Wrap with ActionMasker to enable action masking
env = ActionMasker(env, mask_fn)

# MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
# with ActionMasker. If the wrapper is detected, the masks are automatically
# retrieved and used when learning.

print(torch.cuda.is_available())

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_log_dir = f"logs/tb/{timestamp}"

model = MaskablePPO(
    MultiInputPolicy,
    env,
    verbose=1,
    ent_coef=0.03,
    learning_rate=1e-4,
    n_epochs=4,
    clip_range=0.2,
    tensorboard_log=tensorboard_log_dir
)


model.learn(
    total_timesteps=N_EPISODES * STEPS_PER_EPISODE,
    tb_log_name="ppo_mask_run_{}".format(timestamp)
)

# Get timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"trained_models/init-placement/ppo_mask_{timestamp}"

# Save model
model.save(model_path)
print(f"Model saved to: {model_path}")

# Note that use of masks is manual and optional outside of learning,
# so masking can be "removed" at testing time
# model.predict(observation, action_masks=valid_action_array)