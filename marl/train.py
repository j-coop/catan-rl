import argparse
import os
from functools import partial

import ray
from ray import tune
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPO, PPOConfig

from marl.env.CatanEnv import CatanEnv


def env_creator(config=None):
    return CatanEnv()


def main(num_iterations=2000, stop_timesteps=1_000_000, checkpoint_freq=50):
    ray.init(ignore_reinit_error=True)

    # Wrap PettingZoo AECEnv
    pettingzoo_env = PettingZooEnv(env_creator)

    # Build the policies dict: a single shared policy used for all agents
    # We will let RLlib infer obs/action spaces from a created environment
    # Policy spec: (policy_cls, obs_space, action_space, config)
    def policy_mapping_fn(agent_id, episode, **kwargs):
        # All agents share the same policy name
        return "shared_policy"

    config = (
        PPOConfig()
        .environment(env=PettingZooEnv, env_config={"env_creator": env_creator})
        # Use a small model by default; tune as needed
        .framework("torch")
        .rollouts(num_rollout_workers=1)
        .training(
            gamma=0.99,
            lr=5e-4,
            train_batch_size=4000,
            sgd_minibatch_size=256,
            num_sgd_iter=10,
        )
        .resources(num_gpus=0)
        .debugging(log_level="WARN")
    )

    # Multiagent config: single shared policy for all agents
    config = config.multi_agent(
        policies={"shared_policy": (None, pettingzoo_env.observation_space, pettingzoo_env.action_space, {})},
        policy_mapping_fn=policy_mapping_fn,
        # optionally make observations from other agents available to the policy by customizing here
    )

    # NOTE: RLlib automatically passes obs dicts through; if your observation is a dict
    # with keys 'observation' and 'action_mask', RLlib will include them. With Torch default
    # models, action masking works if the model code checks for 'action_mask' inside the observation.
    #
    # RLlib provides a simple ActionMask wrapper internally when observations include an
    # 'action_mask' key (for newer versions). If you have a custom model, ensure it reads
    # obs["action_mask"] and zeros out logits for masked actions before sampling.

    algo = config.build()

    # training loop
    for i in range(num_iterations):
        result = algo.train()
        print(f"iter={i} reward_mean={result['episode_reward_mean']} timesteps_total={result['timesteps_total']}")

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
