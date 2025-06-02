import gymnasium as gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO, MultiInputPolicy

from envs.base_env.env import CatanBaseEnv
from envs.init_placement_env.env import CatanInitPlacementEnv
from params.catan_constants import N_EPISODES, STEPS_PER_EPISODE


def mask_fn(_env: gym.Env) -> np.ndarray:
    return _env.get_action_masks()

base_env = CatanBaseEnv()
base_env_obs = base_env.reset()

# Create custom env
env = CatanInitPlacementEnv(base_env_obs=base_env_obs)
env.reset()

# Wrap with ActionMasker to enable action masking
env = ActionMasker(env, mask_fn)

# MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
# with ActionMasker. If the wrapper is detected, the masks are automatically
# retrieved and used when learning.
model = MaskablePPO(MultiInputPolicy, env, verbose=1)


model.learn(total_timesteps=N_EPISODES * STEPS_PER_EPISODE)

# Note that use of masks is manual and optional outside of learning,
# so masking can be "removed" at testing time
# model.predict(observation, action_masks=valid_action_array)