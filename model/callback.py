from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os


class AdaptiveLRAndSaveBestCallback(BaseCallback):
    """
    Callback that:
      - Evaluates the agent every `check_freq` steps
      - Saves the best model if mean reward improves
      - Reduces learning rate if reward plateaus
    """

    def __init__(
        self,
        eval_env,
        check_freq=10000,
        patience=5,
        factor=0.5,
        min_lr=1e-6,
        save_path="best_models/",
        n_eval_episodes=5,
        verbose=1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.save_path = save_path
        self.n_eval_episodes = n_eval_episodes

        self.best_mean_reward = -np.inf
        self.no_improve_steps = 0
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            mean_reward = self._evaluate_policy()

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.no_improve_steps = 0
                self._save_best_model()
                if self.verbose:
                    print(f"✅ New best mean reward: {mean_reward:.2f}")
            else:
                # 🔸 No improvement
                self.no_improve_steps += 1
                if self.verbose:
                    print(f"⚠️ No improvement for {self.no_improve_steps} evals.")

                if self.no_improve_steps >= self.patience:
                    self._reduce_lr()
                    self.no_improve_steps = 0

        return True

    def _evaluate_policy(self):
        rewards = []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done, total_reward = False, 0.0
            while not done:
                mask = self.eval_env.unwrapped.get_action_masks()
                action, _states = self.model.predict(obs, deterministic=True,
                                                action_masks=mask)

                obs, reward, done, truncated, info = self.eval_env.step(action)
                done = done or truncated
                total_reward += reward

            rewards.append(total_reward)
        mean_reward = np.mean(rewards)
        if self.verbose:
            print(f"🔍 Evaluation mean reward: {mean_reward:.2f}")
        return mean_reward

    def _reduce_lr(self):
        try:
            current_nominal = float(self.model.lr_schedule(1.0))
        except Exception:
            current_nominal = float(self.model.policy.optimizer.param_groups[0]['lr'])

        new_lr = max(current_nominal * self.factor, self.min_lr)

        # 1) Update optimizer param groups (actual lr used in updates)
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = new_lr

        # 2) Replace the model.lr_schedule so SB3 won't reset it on next update
        self.model.lr_schedule = (lambda lr: (lambda _ : lr))(new_lr)
        # alternative clearer: self.model.lr_schedule = lambda _: new_lr

        if self.verbose:
            print(f"🔽 Learning rate reduced: {current_nominal:.6f} → {new_lr:.6f}")
            self.logger.record("train/learning_rate", new_lr)


    def _save_best_model(self):
        path = os.path.join(self.save_path, f"best_model_1.12.zip")
        self.model.save(path)
        if self.verbose:
            print(f"💾 Best model saved to {path}")
