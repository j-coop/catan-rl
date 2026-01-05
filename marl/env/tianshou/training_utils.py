import os
import torch
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

    def prepare_dict_for_logging(self, data):
        return data

    def write(self, step_type, step, data):
        pass

    def finalize(self):
        pass

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
