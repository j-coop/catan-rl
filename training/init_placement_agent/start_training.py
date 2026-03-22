from datetime import datetime
import torch

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO, MultiInputPolicy

from envs.init_placement_env.road_wrapper import CatanRoadPlacementEnv
from envs.init_placement_env.settlement_wrapper import CatanSettlementPlacementEnv
from params.catan_constants import INIT_PLACEMENT_ENV_N_TIMESTEPS
from .common import *


if __name__ == "__main__":
    # -------------------------------
    # Environment setup
    # -------------------------------
    env = CatanSettlementPlacementEnv()
    env.reset()
    env = ActionMasker(env, mask_fn)

    eval_env = CatanSettlementPlacementEnv(train=False)
    eval_env.reset()
    eval_env = ActionMasker(eval_env, mask_fn)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log_dir = f"logs/tb/{timestamp}"

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],
            vf=[256, 256, 128]
        )
    )

    model = MaskablePPO(
        MultiInputPolicy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        ent_coef=0.01,
        learning_rate=2e-4,
        n_epochs=10,
        clip_range=0.15,
        gae_lambda=0.92,
        n_steps=1024,
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
                 prefix="init_placement_env_1.21",
                 eval_env=eval_env)

    save_final_model(model)
