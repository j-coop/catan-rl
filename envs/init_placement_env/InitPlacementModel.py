import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.base_env.env import CatanBaseEnv
from envs.init_placement_env.env import CatanInitPlacementEnv
from marl.adapters.game_to_base_env import game_to_base_env_state
from marl.model.CatanBoard import CatanBoard
from params.edges_list import EDGES_LIST
from params.tiles2nodes_adjacency_map import TILES_TO_NODES


def mask_fn(env) -> np.ndarray:
    return env.get_action_masks()

"""
Init placement model adapter class
"""
class InitPlacementModel:
    def __init__(self, model_path: str, board: CatanBoard):
        self.model = MaskablePPO.load(model_path)
        self.board = board

    def generate_initial_board(self):
        """
        Returns base_env_obs AFTER all placements
        """
        # Base state from CatanGame
        base_state = game_to_base_env_state(self.board)

        # Create base env
        base_env = CatanBaseEnv(save_env=False, initial_state=base_state)
        base_obs = base_env.reset()

        # Create init placement env
        placement_env = CatanInitPlacementEnv(
            base_env_obs=base_obs,
            train=False,
        )
        placement_env = ActionMasker(placement_env, mask_fn)

        obs, _ = placement_env.reset()

        # Run placement loop
        for _ in range(16):
            mask = placement_env.unwrapped.get_action_masks()
            action, _ = self.model.predict(
                obs,
                deterministic=True,
                action_masks=mask,
            )
            obs, _, _, _, info = placement_env.step(action)

        # base_obs inside env is now mutated
        return placement_env.unwrapped._base_obs

    @staticmethod
    def apply_base_obs_to_game(base_obs, game, player_order=None):
        """
        Apply final base_obs from init placement env to CatanGame.

        base_obs["nodes_owners"]: shape (N_TILES, 6, N_PLAYERS)
        base_obs["edges_owners"]: shape (N_TILES, 6, N_PLAYERS)
        player_order: optional list mapping base_obs player index → CatanGame player name
        """

        if player_order is None:
            player_order = [p.name for p in game.players]

        # Build TILE_NODE_MAP
        TILE_NODE_MAP = {}
        for tile_id, nodes in TILES_TO_NODES.items():
            for local_node_id, global_node_id in enumerate(nodes):
                TILE_NODE_MAP[(tile_id, local_node_id)] = global_node_id

        # Build TILE_EDGE_MAP
        TILE_EDGE_MAP = {}
        for edge_id, (node_a, node_b) in enumerate(EDGES_LIST):
            for tile_id, tile_nodes in TILES_TO_NODES.items():
                for i in range(len(tile_nodes)):
                    n1, n2 = tile_nodes[i], tile_nodes[(i + 1) % 6]
                    if (node_a, node_b) == (n1, n2) or (node_a, node_b) == (n2, n1):
                        TILE_EDGE_MAP[(tile_id, i)] = edge_id
                        break

        # Collect settlements to build
        settlements_to_build = set()
        nodes_owners = base_obs["nodes_owners"]  # shape (N_TILES, 6, N_PLAYERS)
        tiles, local_nodes, n_players = nodes_owners.shape

        for tile_id in range(tiles):
            for local_node_id in range(local_nodes):
                for player_idx in range(n_players):
                    if nodes_owners[tile_id, local_node_id, player_idx]:
                        global_node_id = TILE_NODE_MAP[(tile_id, local_node_id)]
                        settlements_to_build.add((player_order[player_idx], global_node_id))

        # Apply settlements
        for player_name, node_id in settlements_to_build:
            game.build_settlement(player_name, node_id, init_placement=True)

        # Collect roads to build
        roads_to_build = set()
        edges_owners = base_obs["edges_owners"]  # shape (N_TILES, 6, N_PLAYERS)
        tiles, local_edges, n_players = edges_owners.shape

        for tile_id in range(tiles):
            for local_edge_id in range(local_edges):
                for player_idx in range(n_players):
                    if edges_owners[tile_id, local_edge_id, player_idx]:
                        global_edge_id = TILE_EDGE_MAP[(tile_id, local_edge_id)]
                        roads_to_build.add((player_order[player_idx], global_edge_id))

        # Apply roads
        for player_name, edge_id in roads_to_build:
            game.build_road(player_name, edge_id, init_placement=True)

