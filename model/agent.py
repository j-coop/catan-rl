import gymnasium as gym
from datetime import datetime
import torch
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO, MultiInputPolicy

from envs.init_placement_env.env import CatanInitPlacementEnv
from envs.init_placement_env.road_wrapper import CatanRoadPlacementEnv
from envs.init_placement_env.settlement_wrapper import CatanSettlementPlacementEnv
from model.callback import AdaptiveLRAndSaveBestCallback
from params.catan_constants import N_EPISODES, STEPS_PER_EPISODE, EVAL_FREQ, PATIENCE


def mask_fn(_env: gym.Env):
    return _env.get_action_masks()


# ---------------------- Setup ----------------------
print("CUDA available:", torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_log_dir = f"logs/tb/{timestamp}"

# Shared core environment
core_env = CatanInitPlacementEnv()

# Phase-specific environments
settlement_env = ActionMasker(CatanSettlementPlacementEnv(core_env), mask_fn)
road_env = ActionMasker(CatanRoadPlacementEnv(core_env), mask_fn)

# Callback
adaptive_lr_callback = AdaptiveLRAndSaveBestCallback(
    eval_env=settlement_env,  # or road_env if you prefer
    check_freq=EVAL_FREQ,
    patience=PATIENCE,
    factor=0.5,
    min_lr=1e-6,
    save_path="trained_models/best/",
    n_eval_episodes=5,
    verbose=1
)

# ---------------------- Models ----------------------
settlement_model = MaskablePPO(
    MultiInputPolicy,
    settlement_env,
    verbose=1,
    ent_coef=0.1,
    learning_rate=5e-4,
    n_epochs=4,
    clip_range=0.15,
    gae_lambda=0.9,
    n_steps=2048,
    gamma=0.97,
    normalize_advantage=True,
    tensorboard_log=tensorboard_log_dir,
    device=device
)

road_model = MaskablePPO(
    MultiInputPolicy,
    road_env,
    verbose=1,
    ent_coef=0.1,
    learning_rate=5e-4,
    n_epochs=4,
    clip_range=0.15,
    gae_lambda=0.9,
    n_steps=2048,
    gamma=0.97,
    normalize_advantage=True,
    tensorboard_log=tensorboard_log_dir,
    device=device
)

# ---------------------- Synchronized Training ----------------------
from .synchronized_learn import synchronized_learn
from stable_baselines3.common.logger import configure

log_path = "logs/"
logger = configure(log_path, ["stdout", "tensorboard"])
settlement_model.set_logger(logger)
road_model.set_logger(logger)


synchronized_learn(
    settlement_model=settlement_model,
    road_model=road_model,
    settlement_env=settlement_env,
    road_env=road_env,
    total_episodes=N_EPISODES,
    steps_per_episode=STEPS_PER_EPISODE,
    callback=adaptive_lr_callback
)

# ---------------------- Save Models ----------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
settlement_model.save(f"trained_models/init-placement/ppo_settlement_{timestamp}")
road_model.save(f"trained_models/init-placement/ppo_road_{timestamp}")
print(f"Models saved to trained_models/init-placement/")
