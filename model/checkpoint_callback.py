from stable_baselines3.common.callbacks import BaseCallback
import os


class CleanCheckpointCallback(BaseCallback):

    def __init__(self, save_path, save_freq, prefix, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.prefix = prefix
        self.last_checkpint_steps = 0
        os.makedirs(save_path, exist_ok=True)

    def _checkpoint_path(self, timesteps) -> str:
        return os.path.join(self.save_path,
                            f"{self.prefix}_checkpoint_{timesteps}_steps.zip")
    
    def _remove_last_checkpoint(self):
        path = self._checkpoint_path(self.last_checkpint_steps)
        if os.path.exists(path):
            os.remove(path)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = self._checkpoint_path(self.num_timesteps)
            self.model.save(path)
            self._remove_last_checkpoint()
            self.last_checkpint_steps = self.num_timesteps
            if self.verbose:
                print(f"💾 Checkpoint saved: {path}")
        return True
