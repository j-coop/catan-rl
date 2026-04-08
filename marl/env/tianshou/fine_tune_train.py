"""
Phase 2 Fine-Tuning Training Script.

Loads a Phase 1 base model checkpoint and continues training it against a
history pool of former model snapshots and heuristic bots, with a win/loss
terminal reward that makes winning the dominant objective.

Usage:
    python -m marl.env.tianshou.fine_tune_train --base-checkpoint path/to/checkpoint.pt

Phase 1 training is completely unaffected.
"""

import argparse
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.trainer import OnPolicyTrainerParams, OnPolicyTrainer
from torch.distributions import Categorical
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.data.collector import Collector
from tianshou.env import DummyVectorEnv

from marl.env.tianshou.fine_tune_env import FineTuneCatanEnv
from marl.env.tianshou.history_pool import HistoryPool
from marl.env.tianshou.multi_agent_env import CatanEnv
from marl.env.tianshou.actor import MaskedActor
from marl.env.tianshou.critic import Critic
from marl.env.tianshou.training_utils import (
    CheckpointLogger,
    CheckpointManager,
    PPOWithTensorboard,
)
from params.catan_constants import GAMMA


# ==========================================
# HYPERPARAMETERS
# ==========================================
LEARNING_RATE = 5e-5        # Lower than Phase 1 — fine-tuning, not from scratch
WEIGHT_DECAY = 1e-5
ENTROPY_COEF = 0.01         # Low: policy already converged, don't re-randomize
MAX_GRAD_NORM = 0.5
GAE_LAMBDA = 0.95
BATCH_SIZE = 2048
EPOCH_NUM_STEPS = 32_000
MAX_EPOCHS = 500

POOL_CHECKPOINT_DIR = "trained_models/checkpoints"   # Phase 1 checkpoint dir
SAVE_DIR = "trained_models/checkpoints_ft"
LOG_DIR = "logs_ft"
POOL_REFRESH_INTERVAL = 10  # Epochs between pool refreshes
MAX_POOL_SNAPSHOTS = 20
# ==========================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        required=True,
        help="Path to Phase 1 checkpoint to start fine-tuning from.",
    )
    parser.add_argument(
        "--pool-dir",
        type=str,
        default=POOL_CHECKPOINT_DIR,
        help="Directory with Phase 1 checkpoints to use as history pool opponents.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=SAVE_DIR,
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=LOG_DIR,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Dimensions from a scratch env ----
    _tmp = CatanEnv()
    obs_dim = _tmp.observation_spaces["Blue Player"]["observation"].shape[0]
    act_dim = _tmp.action_spaces["Blue Player"].n
    del _tmp

    # ---- History Pool ----
    pool = HistoryPool(
        checkpoint_dir=args.pool_dir,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        max_snapshots=MAX_POOL_SNAPSHOTS,
    )
    pool.refresh()
    print(f"Pool initialized: {pool}")

    # ---- Environment ----
    train_envs = DummyVectorEnv([
        lambda: FineTuneCatanEnv(pool=pool)
    ])

    # ---- Actor / Critic: load Phase 1 weights ----
    actor = MaskedActor(obs_dim, act_dim).to(device)
    critic = Critic(obs_dim).to(device)

    ckpt = torch.load(args.base_checkpoint, map_location=device)
    # ckpt["policy"] is ProbabilisticActorPolicy.state_dict() → keys have "actor." prefix
    actor_state = {
        k.removeprefix("actor."): v
        for k, v in ckpt["policy"].items()
        if k.startswith("actor.")
    }
    actor.load_state_dict(actor_state)
    critic.load_state_dict(ckpt["critic"])
    # Deliberately skip ckpt["optim"] — starting with fresh low LR optimizer
    print(f"Loaded base checkpoint: {args.base_checkpoint}")

    # ---- Policy / Algorithm ----
    policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=lambda logits: Categorical(logits=logits),
        action_space=train_envs.action_space[0],
        action_scaling=False,
    )

    optimizer_factory = AdamOptimizerFactory(
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    algo = PPOWithTensorboard(
        policy=policy,
        critic=critic,
        optim=optimizer_factory,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        max_grad_norm=MAX_GRAD_NORM,
        ent_coef=ENTROPY_COEF,
    )
    # Override the hardcoded "logs" writer with fine-tune log dir
    algo.writer.close()
    algo.writer = SummaryWriter(args.log_dir)
    algo.log_dir = args.log_dir

    # ---- Collector / Logger / Checkpointing ----
    collector = Collector(
        policy=algo.policy,
        env=train_envs,
    )

    checkpoint_manager = CheckpointManager(algo, save_dir=args.save_dir)
    checkpoint_logger = CheckpointLogger(log_dir=args.log_dir)

    def train_fn(epoch, env_step):
        if epoch % POOL_REFRESH_INTERVAL == 0:
            pool.refresh()
            print(f"[Epoch {epoch}] Pool refreshed: {pool}")

    params = OnPolicyTrainerParams(
        training_collector=collector,
        max_epochs=MAX_EPOCHS,
        epoch_num_steps=EPOCH_NUM_STEPS,
        batch_size=BATCH_SIZE,
        save_checkpoint_fn=checkpoint_manager,
        logger=checkpoint_logger,
        training_fn=train_fn,
    )

    result = OnPolicyTrainer(
        algorithm=algo,
        params=params,
    ).run()

    print("Fine-tuning done")
