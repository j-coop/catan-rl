from datetime import datetime
import torch

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO, MultiInputPolicy

from envs.init_placement_env.env import CatanInitPlacementEnv
from params.catan_constants import INIT_PLACEMENT_ENV_N_TIMESTEPS
from .common import *


if __name__ == "__main__":
    # -------------------------------
    # Environment setup
    # -------------------------------
    env = CatanInitPlacementEnv()
    env.reset()
    env = ActionMasker(env, mask_fn)

    eval_env = CatanInitPlacementEnv()
    eval_env.reset()
    eval_env = ActionMasker(eval_env, mask_fn)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log_dir = f"logs/tb/{timestamp}"

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

    # -------------------------------
    # Start training
    # -------------------------------
    run_training(model=model,
                 timesteps=INIT_PLACEMENT_ENV_N_TIMESTEPS,
                 prefix="init_placement_env_1.12",
                 eval_env=eval_env)

    save_final_model(model)
