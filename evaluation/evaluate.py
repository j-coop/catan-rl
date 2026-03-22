import os
import time
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.init_placement_env.env import CatanInitPlacementEnv
from envs.init_placement_env.road_wrapper import CatanRoadPlacementEnv
from envs.init_placement_env.settlement_wrapper import CatanSettlementPlacementEnv
from visualization.map_plotter import CatanMapPlotter


def mask_fn(env) -> np.ndarray:
    return env.get_action_masks()


road_model = MaskablePPO.load(
    "C:/PG/sem_8/PB/trained_models/init-placement/ppo_mask_20260224_233542"
)

settlement_model = MaskablePPO.load(
    "C:/PG/sem_8/PB/trained_models/init-placement/ppo_mask_20260224_234127"
)


# ===============================
# Create ONE shared base state
# ===============================
base_env = CatanInitPlacementEnv(
    ep_done_previously=0,
    base_env_obs=None,
    train=False
)

base_obs, _ = base_env.reset()
shared_state = base_env._base_obs  # authoritative game state


# ===============================
# Create wrappers sharing SAME state
# ===============================
settlement_env = ActionMasker(
    CatanSettlementPlacementEnv(
        ep_done_previously=0,
        base_env_obs=shared_state,
        train=False
    ),
    mask_fn
)

road_env = ActionMasker(
    CatanRoadPlacementEnv(
        ep_done_previously=0,
        base_env_obs=shared_state,
        train=False,
        evaluation=True
    ),
    mask_fn
)

# ===============================
# Initialize environments (DO NOT reset base state again)
# ===============================
settlement_env.reset()
road_env.reset()

timestamp = time.strftime("%Y%m%d-%H%M%S")
save_dir = f"placement_runs/{timestamp}"
os.makedirs(save_dir, exist_ok=True)


for placement_step in range(16):
    if placement_step % 2 == 0:
        # ----- Settlement -----
        env = settlement_env
        model = settlement_model
    else:
        # ----- Road -----
        settlement_id = settlement_env.unwrapped.last_settlement_node_index
        settlement_player = settlement_env.unwrapped.turn_order[settlement_env.unwrapped.turn_index - 1]
        env = road_env
        model = road_model
        env.unwrapped.update_road_placement_mask(
            settlement_id,
            settlement_player
        )

    obs = env.unwrapped._obs
    mask = env.unwrapped.get_action_masks()
    action, _ = model.predict(
        obs,
        deterministic=True,
        action_masks=mask
    )

    print(f"Step {placement_step}: action={action}")
    obs, reward, done, truncated, info = env.step(action)

    if placement_step % 2 == 1:
        filename = os.path.join(
            save_dir,
            f"step_{placement_step:02d}.png"
        )
        plotter = CatanMapPlotter(shared_state)
        plotter.plot_catan_map(filename)
