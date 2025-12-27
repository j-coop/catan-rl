from marl.env.tianshou.multi_agent_env import CatanEnv
from .actor import MaskedActor
from .critic import Critic

from tianshou.trainer import OnPolicyTrainerParams
from tianshou.trainer import OnPolicyTrainer
from tianshou.algorithm.modelfree.ppo import PPO
from torch.distributions import Categorical
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.data.collector import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
import numpy as np


train_envs = DummyVectorEnv([
    lambda: PettingZooEnv(CatanEnv())
])

obs_dim = train_envs.observation_space[0]["observation"].shape[0]
act_dim = train_envs.action_space[0].n

critic = Critic(obs_dim)
actor = MaskedActor(obs_dim, act_dim)
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

algo = PPO(
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

params = OnPolicyTrainerParams(
    training_collector=collector,
    epoch_num_steps=30000,
    batch_size=1024,
    multi_agent_return_reduction=lambda returns: np.array([np.mean(r) if isinstance(r, (list, np.ndarray)) else r for r in returns])
)

result = OnPolicyTrainer(
    algorithm=algo,
    params=params
).run()

print("Training done")
