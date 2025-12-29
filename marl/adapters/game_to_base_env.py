import numpy as np

from marl.model.CatanBoard import CatanBoard
from params.catan_constants import N_TILES, N_TILE_TYPES, N_TOKEN_VALUES, TILE_TYPES, TOKENS, N_PORT_TYPES, N_PLAYERS


def game_to_base_env_state(board: CatanBoard) -> dict:

    resources = np.zeros((N_TILES, N_TILE_TYPES), dtype=np.int8)
    tokens = np.zeros((N_TILES, N_TOKEN_VALUES), dtype=np.int8)
    has_robber = np.zeros(N_TILES, dtype=np.int8)

    for i, (res, token) in enumerate(board.tiles):
        resources[i, TILE_TYPES.index(res)] = 1
        if token is not None:
            tokens[i, TOKENS.index(token)] = 1

    has_robber[board.robber_position] = 1

    return {
        "resources": resources,
        "tokens": tokens,
        "has_robber": has_robber,
        "nodes_settlements": np.zeros((N_TILES, 6), dtype=np.int8),
        "nodes_cities": np.zeros((N_TILES, 6), dtype=np.int8),
        "nodes_owners": np.zeros((N_TILES, 6, N_PLAYERS), dtype=np.int8),
        "nodes_ports": np.zeros(([N_TILES, 6, N_PORT_TYPES])),  # no ports - life is too short to encode list of ports on nodes as [N_TILES, 6, N_PORT_TYPES]
        "edges_owners": np.zeros((N_TILES, 6, N_PLAYERS), dtype=np.int8),
    }
