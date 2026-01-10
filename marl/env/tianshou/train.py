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
from .actor import MaskedActor
from .critic import Critic
from .training_utils import (CheckpointLogger,
                             CheckpointManager,
                             ScalarRewardPettingZooEnv,
                             PPOWithTensorboard)


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
        lr=5e-4,
        weight_decay=1e-4,
    )

    algo = PPOWithTensorboard(
        policy=policy,
        critic=critic,
        optim=optimizer_factory,
        gamma=0.97,
        gae_lambda=0.95,
        max_grad_norm=0.5,
    )

    collector = Collector(
        policy=algo.policy,
        env=train_envs
    )

    checkpoint_manager = CheckpointManager(algo)
    checkpoint_logger= CheckpointLogger()

    params = OnPolicyTrainerParams(
        training_collector=collector,
        max_epochs=100,
        epoch_num_steps=64_000,
        batch_size=1024,
        save_checkpoint_fn=checkpoint_manager,
        logger=checkpoint_logger
    )

    result = OnPolicyTrainer(
        algorithm=algo,
        params=params
    ).run()

    print("Training done")
