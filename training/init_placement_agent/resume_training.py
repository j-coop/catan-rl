import torch
import re
import os

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from envs.init_placement_env.env import CatanInitPlacementEnv

from params.catan_constants import (INIT_PLACEMENT_ENV_N_EPISODES,
                                    INIT_PLACEMENT_ENV_STEPS_PER_EPISODE)
from .common import *


RESUME_CHECKPOINT_PATH = "trained_models/checkpoints/init_placement_env_1.12_checkpoint_4000_episodes.zip"


def extract_steps_from_checkpoint() -> int:
    """
    Extract the number of steps from a Stable-Baselines3 checkpoint path.
    """
    match = re.search(r"_checkpoint_(\d+)_episodes\.zip$",
                      RESUME_CHECKPOINT_PATH)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(
            f"Cannot extract steps from path: {RESUME_CHECKPOINT_PATH}")

def extract_checkpoint_marker() -> str:
    match = re.search(r"checkpoints/(.+)_checkpoint", RESUME_CHECKPOINT_PATH)
    if match:
        return match.group(1)
    else:
        raise ValueError(
            f"Cannot extract marker from path: {RESUME_CHECKPOINT_PATH}")


if __name__ == "__main__":
    # -------------------------------
    # Environment setup
    # -------------------------------
    episodes_already_done = extract_steps_from_checkpoint()

    env = CatanInitPlacementEnv(ep_done_previously=episodes_already_done)
    env.reset()
    env = ActionMasker(env, mask_fn)

    eval_env = CatanInitPlacementEnv()
    eval_env.reset()
    eval_env = ActionMasker(eval_env, mask_fn)

    # -------------------------------
    # Load model from checkpoint
    # -------------------------------
    if not os.path.exists(RESUME_CHECKPOINT_PATH):
        raise ValueError("Checkpoint file not found")

    print("Loading model from checkpoint:", RESUME_CHECKPOINT_PATH)
    model = MaskablePPO.load(
        RESUME_CHECKPOINT_PATH,
        env=env,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    episodes_remaining = INIT_PLACEMENT_ENV_N_EPISODES - episodes_already_done
    timesteps = episodes_remaining * INIT_PLACEMENT_ENV_STEPS_PER_EPISODE
    prefix = extract_checkpoint_marker()

    eval_env = CatanInitPlacementEnv(ep_done_previously=episodes_already_done)
    eval_env.reset()
    eval_env = ActionMasker(eval_env, mask_fn)

    # -------------------------------
    # Resume training from checkpoint
    # -------------------------------
    print(f"Resuming the training for {episodes_remaining} timesteps")
    run_training(model=model,
                 timesteps=timesteps,
                 prefix=prefix,
                 eval_env=eval_env,
                 ep_done=episodes_already_done)
    save_final_model(model)
