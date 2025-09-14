import gymnasium as gym
import numpy as np
from datetime import datetime
import torch

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO, MultiInputPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs.base_env.env import CatanBaseEnv
from envs.init_placement_env.CatanPolicy import CatanPolicy
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
    # CatanPolicy,
    MultiInputPolicy,
    env,
    verbose=1,
    ent_coef=0.1,
    learning_rate=3e-4,
    n_epochs=4,
    clip_range=0.2,
    gae_lambda=0.97,
    n_steps=4096,
    normalize_advantage=True,
    tensorboard_log=tensorboard_log_dir
)

# Separate env instance for eval callback
eval_base_env = CatanBaseEnv(save_env=True)
eval_base_env_obs = eval_base_env.reset()

# Create placement env and wrap with ActionMasker
eval_env = CatanInitPlacementEnv(base_env_obs=eval_base_env_obs, train=False)
eval_env = Monitor(eval_env)
eval_env = ActionMasker(eval_env, mask_fn)

eval_env.reset()

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"trained_models/init-placement/best_{timestamp}/best_model",
    log_path="logs/eval",
    eval_freq=50000,          # Evaluate every 50000 timesteps
    n_eval_episodes=10,      # Number of episodes per evaluation
    deterministic=True,      # Use deterministic actions during eval
    render=False
)


model.learn(
    total_timesteps=N_EPISODES * STEPS_PER_EPISODE,
    tb_log_name="ppo_mask_run_{}".format(timestamp),
    # callback=eval_callback
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