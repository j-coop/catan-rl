import gymnasium as gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from envs.init_placement_env.env import CatanInitPlacementEnv
from params.catan_constants import N_EPISODES, STEPS_PER_EPISODE


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()

base_env_obs = None

# Create custom env
env = CatanInitPlacementEnv(base_env_obs=base_env_obs)
# Wrap with ActionMasker to enable action masking
env = ActionMasker(env, mask_fn)

# MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
# with ActionMasker. If the wrapper is detected, the masks are automatically
# retrieved and used when learning.
model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)


model.learn(total_timesteps=N_EPISODES * STEPS_PER_EPISODE)

# Note that use of masks is manual and optional outside of learning,
# so masking can be "removed" at testing time
# model.predict(observation, action_masks=valid_action_array)