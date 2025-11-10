import numpy as np
import torch
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.buffers import RolloutBuffer


def synchronized_learn(settlement_model: MaskablePPO,
                       road_model: MaskablePPO,
                       settlement_env,
                       road_env,
                       total_episodes: int,
                       steps_per_episode: int,
                       callback=None):
    """
    Custom learn function for synchronized settlement+road PPO training
    """
    # Ensure models have rollout buffers
    if settlement_model.rollout_buffer is None:
        settlement_model.rollout_buffer = RolloutBuffer(
            buffer_size=settlement_model.n_steps,
            observation_space=settlement_env.observation_space,
            action_space=settlement_env.action_space,
            device=settlement_model.device,
            n_envs=1
        )
    if road_model.rollout_buffer is None:
        road_model.rollout_buffer = RolloutBuffer(
            buffer_size=road_model.n_steps,
            observation_space=road_env.observation_space,
            action_space=road_env.action_space,
            device=road_model.device,
            n_envs=1
        )

    for ep in range(total_episodes):
        obs, _ = settlement_env.reset()
        done = False

        steps_collected = 0
        while steps_collected < road_model.n_steps:

            for step in range(steps_per_episode):
                print(f"Step {ep}, buffer pos: {settlement_model.rollout_buffer.pos}, full: {settlement_model.rollout_buffer.full}")
                # ================== Settlement Phase ==================
                s_mask = settlement_env.action_masks()
                s_action, s_state = settlement_model.predict(obs, action_masks=s_mask, deterministic=False)
                obs_next, s_reward, s_done, _, _ = settlement_env.step(s_action)

                # Add to settlement buffer
                device = settlement_model.device
                settlement_model.rollout_buffer.add(
                    obs=obs,
                    action=s_action,
                    reward=torch.tensor([s_reward], dtype=torch.float32, device=device),
                    episode_start=torch.tensor([False], dtype=torch.bool, device=device),
                    value=torch.tensor([0.0], dtype=torch.bool, device=device),
                    log_prob=torch.tensor([0.0], dtype=torch.bool, device=device)
                )

                # ================== Road Phase ========================
                r_mask = road_env.action_masks()
                r_action, r_state = road_model.predict(obs_next, action_masks=r_mask, deterministic=False)
                obs_next, r_reward, done, _, _ = road_env.step(r_action)

                # Add to road buffer
                device = road_model.device
                road_model.rollout_buffer.add(
                    obs=obs_next,
                    action=r_action,
                    reward=torch.tensor([r_reward], dtype=torch.float32, device=device),
                    episode_start=torch.tensor([False], dtype=torch.bool, device=device),
                    value=torch.tensor([0.0], dtype=torch.bool, device=device),
                    log_prob=torch.tensor([0.0], dtype=torch.bool, device=device)
                )

                obs = obs_next  # update current obs

                # ================== Callback =========================
                if callback is not None:
                    callback.num_timesteps += 1
                    callback.n_calls += 1
                    if not callback._on_step():
                        return  # callback requested stop

                steps_collected += 1

                if done:
                    break

        # ================== Update Policies ======================
        settlement_model.train()
        road_model.train()
