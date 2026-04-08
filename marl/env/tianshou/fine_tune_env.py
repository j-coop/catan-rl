"""
Fine-tuning environment: wraps CatanEnv (AEC multi-agent) as a single-agent
gymnasium.Env from Tianshou's perspective.

One fixed agent ("Blue Player") is the trainee. The other three agents are
controlled internally by policies sampled from the HistoryPool. Tianshou only
ever sees the trainee's observations and rewards.

Terminal win/loss bonuses are added here, outside the ±7 PBRS clip, so that
winning is a strong signal that dominates reward shaping.
"""

import gymnasium
import numpy as np
from gymnasium import spaces

from marl.env.tianshou.multi_agent_env import CatanEnv
from marl.env.tianshou.history_pool import HistoryPool

TRAINEE = "Blue Player"
WIN_REWARD_FT = 50.0
LOSS_PENALTY_FT = -25.0


class FineTuneCatanEnv(gymnasium.Env):
    """
    Single-agent gymnasium wrapper around CatanEnv for Phase 2 fine-tuning.

    The trainee is always "Blue Player" (index 0 in the agent list).
    On each reset(), three opponents are sampled from the HistoryPool.
    Whenever it is not the trainee's turn, opponents are stepped automatically.
    """

    metadata = {"render_modes": []}

    def __init__(self, pool: HistoryPool):
        super().__init__()
        self.pool = pool
        self._inner = CatanEnv()
        self.opponent_policies = []

        inner_obs = self._inner.observation_spaces[TRAINEE]
        obs_dim = inner_obs["observation"].shape[0]
        act_dim = inner_obs["action_mask"].shape[0]
        # Match PettingZooEnv's "obs"/"mask" key format so that MaskedActor
        # (which reads obs.obs and obs.mask) and Critic (obs.obs) work unchanged.
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32),
            "mask": spaces.Box(low=0, high=1, shape=(act_dim,), dtype=np.int8),
        })
        self.action_space = self._inner.action_spaces[TRAINEE]

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        self._inner.reset(seed=seed, options=options)
        self.opponent_policies = self.pool.sample_opponents(n=3)
        self._advance_to_trainee()
        return self._wrap_obs(self._inner.observe(TRAINEE)), {}

    def step(self, action: int):
        # Execute trainee's action
        self._inner.step(action)
        step_reward = float(self._inner.rewards.get(TRAINEE, 0.0))

        # Auto-step opponents until it's the trainee's turn (or game over)
        terminated = self._advance_to_trainee()

        if terminated:
            winner = self._inner.game.winner
            step_reward += WIN_REWARD_FT if winner == TRAINEE else LOSS_PENALTY_FT

        return self._wrap_obs(self._inner.observe(TRAINEE)), step_reward, terminated, False, {}

    def render(self):
        pass

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_obs(obs_dict: dict) -> dict:
        """
        Convert {"observation": array, "action_mask": array}
        → {"obs": array, "mask": array}
        to match PettingZooEnv's format that MaskedActor/Critic expect.
        """
        return {
            "obs": obs_dict["observation"],
            "mask": obs_dict["action_mask"],
        }

    def _advance_to_trainee(self) -> bool:
        """
        Step opponents one at a time until it's the trainee's turn or the game ends.
        Returns True if the game is over.
        """
        while (
            not self._inner.game.game_over
            and self._inner.agent_selection != TRAINEE
        ):
            agent = self._inner.agent_selection
            obs_dict = self._inner.observe(agent)
            action = self._get_opponent_action(agent, obs_dict)
            self._inner.step(action)

        return self._inner.game.game_over

    def _get_opponent_action(self, agent: str, obs_dict: dict) -> int:
        """
        Select an action for an opponent agent using their sampled policy.
        opponent_policies[i] corresponds to the i-th non-trainee agent slot
        (Purple=0, Yellow=1, Green=2 from trainee's perspective).
        """
        non_trainee = [a for a in self._inner.possible_agents if a != TRAINEE]
        try:
            slot = non_trainee.index(agent)
            policy = self.opponent_policies[slot]
        except (ValueError, IndexError):
            # Fallback: random valid action
            mask = obs_dict["action_mask"]
            valid = np.where(mask == 1)[0]
            return int(np.random.choice(valid))

        return policy.select_action(obs_dict)
