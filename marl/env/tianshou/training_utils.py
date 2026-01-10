import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.env import PettingZooEnv
from tianshou.utils.logger.logger_base import BaseLogger
import numpy as np


class ScalarRewardPettingZooEnv(PettingZooEnv):
    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)
        if isinstance(rew, (list, tuple, np.ndarray)):
            rew = float(np.sum(rew))
        return obs, rew, terminated, truncated, info


class CheckpointManager:

    def __init__(self, algo, save_dir="trained_models\checkpoints"):
        self.algo = algo
        self.save_dir = save_dir
        self.last_epoch_saved = -1
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, epoch: int, env_step: int, gradient_step: int) -> str:
        # Save only once per epoch
        if epoch == self.last_epoch_saved:
            return ""

        self.last_epoch_saved = epoch

        path = os.path.join(
            self.save_dir,
            f"ppo_catan_{epoch}.pt"
        )

        torch.save(
            {
                "epoch": epoch,
                "env_step": env_step,
                "gradient_step": gradient_step,
                "policy": self.algo.policy.state_dict(),
                "critic": self.algo.critic.state_dict(),
                "optim": self.algo.optim.state_dict(),
            },
            path
        )
        return path
    

class CheckpointLogger(BaseLogger):

    def __init__(self, log_dir="logs"):
        super().__init__()
        self.writer = SummaryWriter(log_dir)

    def prepare_dict_for_logging(self, data):
        log_data = {}

        if 'returns_stat' in data:
            log_data['reward_mean'] = float(data['returns_stat']['mean'])
            log_data['reward_std'] = float(data['returns_stat']['std'])
            log_data['reward_max'] = float(data['returns_stat']['max'])
            log_data['reward_min'] = float(data['returns_stat']['min'])

        if 'lens_stat' in data:
            log_data['episode_len_mean'] = float(data['lens_stat']['mean'])
            log_data['episode_len_std'] = float(data['lens_stat']['std'])
            log_data['episode_len_max'] = float(data['lens_stat']['max'])
            log_data['episode_len_min'] = float(data['lens_stat']['min'])

        if 'pred_dist_std_array_stat' in data and 0 in data['pred_dist_std_array_stat']:
            log_data['policy_std_mean'] = float(data['pred_dist_std_array_stat'][0]['mean'])
            log_data['policy_std_max'] = float(data['pred_dist_std_array_stat'][0]['max'])

        return log_data

    def write(self, step_type, step, data):
        for key, value in data.items():
            if isinstance(value, (float, int, np.floating, np.integer)):
                print( f"Logging {key} : {value} at step {step}")
                self.writer.add_scalar(key, value, step)

    def finalize(self):
        self.writer.close()

    def save_data(
        self,
        epoch: int,
        env_step: int,
        update_step: int,
        save_checkpoint_fn=None,
    ) -> None:
        if save_checkpoint_fn is not None:
            save_checkpoint_fn(epoch, env_step, update_step)
            print(f"The checkpoint for epoch {epoch} has been saved.")

    def restore_data(self):
        return 0, 0, 0

    @staticmethod
    def restore_logged_data(log_path: str) -> dict:
        return {}


class PPOWithTensorboard(PPO):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_dir="logs"
        self.global_step = 0
        self.writer = SummaryWriter(self.log_dir)


    def _update_with_batch(self, batch, batch_size=None, repeat=1):
        stats = super()._update_with_batch(batch, batch_size, repeat)

        # Prepare metrics dictionary
        metrics = {
            "loss_actor": float(stats.actor_loss.mean),
            "loss_critic": float(stats.vf_loss.mean),
            "entropy": float(stats.ent_loss.mean),
            "total_loss": float(stats.loss.mean),
            "grad_steps": stats.gradient_steps
        }

        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.global_step)
        self.writer.flush()
        self.global_step += 1

        return stats


def load_checkpoint(path: str, algo: PPO):
    """
    Load a checkpoint for PPO algorithm (actor, critic, optimizer).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint {path} not found")

    checkpoint = torch.load(path, map_location="cpu")  # use "cuda" if needed

    # Load model states
    algo.policy.load_state_dict(checkpoint["policy"])
    algo.critic.load_state_dict(checkpoint["critic"])
    algo.optim.load_state_dict(checkpoint["optim"])

    # Return metadata
    return {
        "epoch": checkpoint.get("epoch", 0),
        "env_step": checkpoint.get("env_step", 0),
        "gradient_step": checkpoint.get("gradient_step", 0),
    }
