import torch

from tianshou.trainer import OnPolicyTrainerParams
from tianshou.trainer import OnPolicyTrainer
from tianshou.algorithm.modelfree.ppo import PPO
from torch.distributions import Categorical
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.data.collector import Collector
from tianshou.env import DummyVectorEnv

from marl.env.tianshou.multi_agent_env import CatanEnv
from marl.env.tianshou.actor import MaskedActor
from marl.env.tianshou.critic import Critic
from marl.env.tianshou.training_utils import (CheckpointLogger,
                                              CheckpointManager,
                                              ScalarRewardPettingZooEnv,
                                              PPOWithTensorboard)
from params.catan_constants import GAMMA

# ==========================================
# HYPERPARAMETERS
# ==========================================
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
ENTROPY_COEF = 0.05  

# PPO Constraints
GAE_LAMBDA = 0.92
MAX_GRAD_NORM = 0.8

# Training lengths
BATCH_SIZE = 2048
EPOCH_NUM_STEPS = 32_000
MAX_EPOCHS = 100
# ==========================================

if __name__ == '__main__':

    train_envs = DummyVectorEnv([
        lambda: ScalarRewardPettingZooEnv(CatanEnv())
    ])

    obs_dim = train_envs.observation_space[0]["observation"].shape[0]
    act_dim = train_envs.action_space[0].n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    critic = Critic(obs_dim).to(device)
    actor = MaskedActor(obs_dim, act_dim).to(device)
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

    collector = Collector(
        policy=algo.policy,
        env=train_envs
    )

    checkpoint_manager = CheckpointManager(algo)
    checkpoint_logger = CheckpointLogger()

    params = OnPolicyTrainerParams(
        training_collector=collector,
        max_epochs=MAX_EPOCHS,
        epoch_num_steps=EPOCH_NUM_STEPS,
        batch_size=BATCH_SIZE,
        save_checkpoint_fn=checkpoint_manager,
        logger=checkpoint_logger
    )

    result = OnPolicyTrainer(
        algorithm=algo,
        params=params
    ).run()

    print("Training done")
