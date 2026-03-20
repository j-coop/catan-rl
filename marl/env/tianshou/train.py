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
from params.catan_constants import (BOARD_LATENT,
                                    FINAL_LATENT,
                                    OTHERS_LATENT,
                                    SELF_LATENT)
from params.catan_constants import (GAMMA,
                                    IS_ENCODER_ENABLED)
from auto_encoder.encoders import CatanFactorizedAutoEncoder


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
MAX_EPOCHS = 1000
# ==========================================

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = None
    if IS_ENCODER_ENABLED:
        encoder = CatanFactorizedAutoEncoder(
            board_latent=BOARD_LATENT,
            self_latent=SELF_LATENT,
            others_latent=OTHERS_LATENT,
            final_latent=FINAL_LATENT
        ).to(device)
        encoder.load_state_dict(
            torch.load(
                "/home/student/Dokumenty/s184725/magisterka/catan-rl/marl/env/tianshou/trained_models/catan_contrastive_lr0.0001_temp0.1.pth",
                map_location=device
            )
        )
        encoder.eval()

        # Freeze encoder
        for p in encoder.parameters():
            p.requires_grad = False

    train_envs = DummyVectorEnv([
        lambda: ScalarRewardPettingZooEnv(CatanEnv())
    ])

    obs_dim = train_envs.observation_space[0]["observation"].shape[0]
    act_dim = train_envs.action_space[0].n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = MaskedActor(obs_dim, act_dim, encoder=encoder).to(device)
    critic = Critic(obs_dim, encoder=encoder).to(device)
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
