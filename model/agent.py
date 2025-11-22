import gymnasium as gym
import numpy as np
from datetime import datetime
import torch

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO, MultiInputPolicy
from stable_baselines3.common.callbacks import CallbackList


from model.adaptive_lr_callback import AdaptiveLRAndSaveBestCallback
from model.checkpoint_callback import CleanCheckpointCallback
from envs.init_placement_env.env import CatanInitPlacementEnv
from params.catan_constants import (INIT_PLACEMENT_ENV_N_TIMESTEPS,
                                    INIT_PLACEMENT_ENV_EVAL_FREQ,
                                    INIT_PLACEMENT_ENV_CHECKPOINT_SAVE_FREQ,
                                    INIT_PLACEMENT_ENV_PATIENCE)


def mask_fn(_env: gym.Env) -> np.ndarray:
    return _env.get_action_masks()

# MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
# with ActionMasker. If the wrapper is detected, the masks are automatically
# retrieved and used when learning.

print(torch.cuda.is_available())

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_log_dir = f"logs/tb/{timestamp}"


# Create custom env
env = CatanInitPlacementEnv()
env.reset()
env = ActionMasker(env, mask_fn)

# Create eval env
eval_env = CatanInitPlacementEnv()
eval_env.reset()
eval_env = ActionMasker(eval_env, mask_fn)

# Create callbacks
adaptive_lr_callback = AdaptiveLRAndSaveBestCallback(
    eval_env=eval_env,
    check_freq=INIT_PLACEMENT_ENV_EVAL_FREQ,
    patience=INIT_PLACEMENT_ENV_PATIENCE,
    factor=0.5,
    min_lr=1e-6,
    save_path="trained_models/best/",
    n_eval_episodes=10,
    verbose=1
)
checkpoint_callback = CleanCheckpointCallback(
    save_freq=INIT_PLACEMENT_ENV_CHECKPOINT_SAVE_FREQ,
    save_path="trained_models/checkpoints/",
    prefix="init_placement_env_1.12"
)

model = MaskablePPO(
    MultiInputPolicy,
    env,
    verbose=1,
    ent_coef=0.1,
    learning_rate=5e-4,
    n_epochs=4,
    clip_range=0.15,
    gae_lambda=0.9,
    n_steps=2048,
    gamma = 0.97,
    normalize_advantage=True,
    tensorboard_log=tensorboard_log_dir,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

model.learn(
    total_timesteps=INIT_PLACEMENT_ENV_N_TIMESTEPS,
    tb_log_name="ppo_mask_run_{}".format(timestamp),
    callback=CallbackList([adaptive_lr_callback, checkpoint_callback]),
    log_interval=10
)

# Get timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"trained_models/init-placement/ppo_mask_{timestamp}"

# Save model
model.save(model_path)
print(f"Model saved to: {model_path}")
