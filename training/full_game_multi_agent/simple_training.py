import os
import logging

from params.catan_constants import GAMMA

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
os.environ["RAY_DEDUP_LOGS"] = "1"
os.environ["RAY_BACKEND_LOG_LEVEL"] = "error"

import ray
import argparse
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from marl.env.CatanEnv import CatanEnv


# ---------------------------------------------------------------------
# Environment registration
# ---------------------------------------------------------------------

def env_creator(config=None):
    return CatanEnv()

register_env("catan", env_creator)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def main(num_iterations=2000, stop_timesteps=1_000_000, checkpoint_freq=50):

    ray.init(ignore_reinit_error=True)

    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.rllib").setLevel(logging.ERROR)
    logging.getLogger("ray.tune").setLevel(logging.ERROR)

    # Create temp env to grab spaces
    temp_env = CatanEnv()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    config = (
        PPOConfig()
        .environment(
            env="catan",
            env_config={}
        )
        # Old, stable API stack (correct for custom MA envs)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .framework("torch")
        .training(
            gamma=GAMMA,
            lr=5e-4,
            train_batch_size=4000,
            num_sgd_iter=10,
        )
        .multi_agent(
            policies={
                "shared_policy": (
                    None,          # default PPO policy
                    obs_space,
                    act_space,
                    {},
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .debugging(log_level="ERROR")
        .env_runners(num_env_runners=1)  # one concurrent worker for logs clarity
    )

    algo = config.build()

    for i in range(num_iterations):
        result = algo.train()
        print(
            f"iter={i} "
            f"result:{result}"
        )

        if i % checkpoint_freq == 0:
            checkpoint = algo.save()
            print("Saved checkpoint to", checkpoint)

        if result["timesteps_total"] >= stop_timesteps:
            break

    algo.stop()
    ray.shutdown()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    args = parser.parse_args()

    main(
        num_iterations=args.iters,
        stop_timesteps=args.timesteps,
    )
