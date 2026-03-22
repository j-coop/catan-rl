from typing import List

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.base_env.env import CatanBaseEnv
from envs.init_placement_env.road_wrapper import CatanRoadPlacementEnv
from envs.init_placement_env.settlement_wrapper import CatanSettlementPlacementEnv
from marl.adapters.game_to_base_env import game_to_base_env_state
from marl.model.CatanBoard import CatanBoard
from params.catan_constants import N_NODES


def mask_fn(env) -> np.ndarray:
    return env.get_action_masks()

"""
Init placement model adapter class
"""
class InitPlacementModel:
    def __init__(
            self,
            settlement_model_path: str,
            road_model_path: str,
            board: CatanBoard
        ):
        self.settlement_model = MaskablePPO.load(settlement_model_path)
        self.road_model = MaskablePPO.load(road_model_path)
        self.board = board

        # Order of placing settlements and roads
        self.settlements_history = []
        self.roads_history = []

    def generate_initial_board(self):
        # Create base env
        base_env = CatanBaseEnv(
            save_env=False,
            initial_state=game_to_base_env_state(self.board)
        )
        base_obs = base_env.reset()

        # Create init placement envs
        settlement_env = ActionMasker(
            CatanSettlementPlacementEnv(
                ep_done_previously=0,
                base_env_obs=base_obs,
                train=False
            ),
            mask_fn
        )

        road_env = ActionMasker(
            CatanRoadPlacementEnv(
                ep_done_previously=0,
                base_env_obs=base_obs,
                train=False,
                evaluation=True
            ),
            mask_fn
        )

        settlement_env.reset()
        road_env.reset()

        for placement_step in range(16):
            if placement_step % 2 == 0:
                # ----- Settlement -----
                env = settlement_env
                model = self.settlement_model
            else:
                # ----- Road -----
                settlement_id = settlement_env.unwrapped.last_settlement_node_index
                settlement_player = settlement_env.unwrapped.turn_order[settlement_env.unwrapped.turn_index - 1]
                env = road_env
                model = self.road_model
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
            player_index = env.unwrapped.turn_order[placement_step // 2]
            if placement_step % 2 == 0:
                self.settlements_history.append((int(action), player_index))
            else:
                self.roads_history.append((int(action) - N_NODES, player_index))
            obs, _, _, _, _ = env.step(action)

    def get_init_roads_and_settlements(self):
        return self.settlements_history, self.roads_history
