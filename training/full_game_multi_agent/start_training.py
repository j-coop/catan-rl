
import os
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import ray
import argparse
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility


from marl.env.CatanEnv import CatanEnv


def env_creator(config=None):
    return PettingZooEnv(CatanEnv())

register_env("catan", env_creator)


def main(num_iterations=2000, stop_timesteps=1_000_000, checkpoint_freq=50):

    ray.init(ignore_reinit_error=True)
    config = (
        PPOConfig()
        .environment(
            env="catan",
            env_config={}
        )
        .framework("torch")
        .training(
            gamma=0.97,
            lr=5e-4,
            train_batch_size=4000,
            num_sgd_iter=10,
        )
        .debugging(log_level="WARN")
    )

    temp_env = CatanEnv()
    policies = {
        "shared_policy": (
            None,
            temp_env.observation_space,
            temp_env.action_space,
            {}
        )
    }
    temp_env.close()

    config = config.multi_agent(
        policies=policies,
        policy_mapping_fn=lambda aid, *args, **kw: "shared_policy"
    )
    algo = config.build()

    for i in range(num_iterations):
        result = algo.train()
        print(f"iter={i} reward_mean={result['episode_reward_mean']}"
              " timesteps_total={result['timesteps_total']}")

        if i % checkpoint_freq == 0:
            checkpoint = algo.save()
            print("Saved checkpoint to", checkpoint)

        if result["timesteps_total"] >= stop_timesteps:
            break

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    args = parser.parse_args()
    main(num_iterations=args.iters, stop_timesteps=args.timesteps)
