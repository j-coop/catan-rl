from envs.base_env.env import CatanBaseEnv
from params.catan_constants import (N_NODES,
                                    N_RESOURCE_TYPES,
                                    TOKENS,
                                    DICE_PROBABILITIES,
                                    NUM_ROLLS,
                                    BEST_EXPECTED_GAIN)
from params.nodes2tiles_adjacency_map import NODES_TO_TILES
import numpy as np


def _simulate_dice_rolls(obs, node_id):
    adjacent_tiles = NODES_TO_TILES[node_id]
    adjacent_tiles_resources = [np.argmax(obs["resources"][tile]) for tile in adjacent_tiles]
    adjacent_tiles_tokens_ids = [np.argmax(obs["tokens"][tile]) for tile in adjacent_tiles]
    adjacent_tiles_tokens = [TOKENS[i] for i in adjacent_tiles_tokens_ids]

    gains = [0 for _ in range(N_RESOURCE_TYPES)]
    for i in range(len(adjacent_tiles_resources)):
        resource = adjacent_tiles_resources[i]
        token = adjacent_tiles_tokens[i]
        expected_gain = DICE_PROBABILITIES[token] * NUM_ROLLS

        if resource < N_RESOURCE_TYPES:  # Ignore desert
            gains[resource] += expected_gain

    # Award reward
    sum_gain = sum(gains)
    normalized_gain_score = sum_gain / BEST_EXPECTED_GAIN
    return normalized_gain_score


N_RUNS = 10000
total_reward = 0.0

for i in range(N_RUNS):
    base_env = CatanBaseEnv(save_env=False)
    obs = base_env.reset()
    board_reward = 0.0
    for j in range(N_NODES):
        reward = _simulate_dice_rolls(obs, j)
        board_reward += reward
    total_reward += board_reward / N_NODES

print(total_reward / N_RUNS)

