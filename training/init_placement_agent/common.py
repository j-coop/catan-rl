import numpy as np
from datetime import datetime

import gymnasium as gym
from stable_baselines3.common.callbacks import CallbackList

from .adaptive_lr_callback import AdaptiveLRAndSaveBestCallback
from .checkpoint_callback import CleanCheckpointCallback
from params.catan_constants import (INIT_PLACEMENT_ENV_EVAL_FREQ,
                                    INIT_PLACEMENT_ENV_CHECKPOINT_SAVE_FREQ,
                                    INIT_PLACEMENT_ENV_PATIENCE,
                                    INIT_PLACEMENT_ENV_STEPS_PER_EPISODE)


def mask_fn(_env: gym.Env) -> np.ndarray:
    return _env.get_action_masks()

def get_adaptive_lr_callback(eval_env):
    adaptive_lr_callback = AdaptiveLRAndSaveBestCallback(
        eval_env=eval_env,
        check_freq=INIT_PLACEMENT_ENV_EVAL_FREQ,
        patience=INIT_PLACEMENT_ENV_PATIENCE,
        factor=0.5,
        min_lr=1e-6,
        save_path="trained_models/best/",
        n_eval_episodes=50,
        verbose=1
    )
    return adaptive_lr_callback

def get_clean_checkpoint_callback(prefix, ep_done=0):
    checkpoint_callback = CleanCheckpointCallback(
        save_freq=INIT_PLACEMENT_ENV_CHECKPOINT_SAVE_FREQ,
        save_path="trained_models/checkpoints/",
        prefix=prefix,
        steps_per_ep=INIT_PLACEMENT_ENV_STEPS_PER_EPISODE,
        ep_done_previously=ep_done
    )
    return checkpoint_callback

def run_training(model, timesteps, prefix, eval_env, ep_done=0):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callback = CallbackList([get_adaptive_lr_callback(eval_env),
                             get_clean_checkpoint_callback(prefix, ep_done)])
    model.learn(
        total_timesteps=timesteps,
        tb_log_name=f"ppo_mask_run_{timestamp}",
        callback=callback,
        log_interval=10
    )

def save_final_model(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"trained_models/init-placement/ppo_mask_{timestamp}"

    model.save(model_path)
    print(f"Model saved to: {model_path}")
