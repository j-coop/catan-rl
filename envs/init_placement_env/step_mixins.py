import numpy as np
from params.catan_constants import *
from params.edges_list import EDGES_LIST
from params.nodes2nodes_adjacency_map import NODES_TO_NODES
from params.nodes2tiles_adjacency_map import NODES_TO_TILES
from params.tiles2nodes_adjacency_map import TILES_TO_NODES


class CatanStepMixin:

    def __build_settlement(self, node_id, player_id):
        for i in range(N_NODES):
            for j in range(len(self._ring_neighbors[i])):
                if node_id == self._ring_neighbors[i][j]:
                    self._obs["adj_is_built"][i][j] = 1

        for tile_id in range(N_TILES):
            adj_nodes = TILES_TO_NODES[tile_id]
            if node_id < adj_nodes[0] or node_id > adj_nodes[-3]:
                continue
            for i in range(len(adj_nodes)):
                if adj_nodes[i] == node_id:
                    self._base_obs["nodes_owners"][tile_id][i][player_id] = 1
                    self._base_obs["nodes_settlements"][tile_id][i] = 1

    def __build_road(self, edge_id, player_id):
        edge_coords = EDGES_LIST[edge_id]
        for tile_id in range(N_TILES):
            adj_nodes = TILES_TO_NODES[tile_id]
            for i in range(len(adj_nodes)):
                if (adj_nodes[i], adj_nodes[(i + 1) % 6]) == edge_coords:
                    self._base_obs["edges_owners"][tile_id][i][player_id] = 1

    def _make_settlement_action(self, player, settlement_action):
        node_id = np.argmax(settlement_action)
        self.__build_settlement(node_id, player)
        self._update_settlement_placement_mask(node_id)
        self._update_road_placement_mask(node_id, player)

    def _make_road_action(self, player, road_action):
        edge_id = np.argmax(road_action)
        self.__build_road(edge_id, player)

    def __get_other_adjacent_nodes(self, node, known_node):
        possible_nodes = NODES_TO_NODES[node].copy()
        possible_nodes.remove(known_node)
        return possible_nodes

    """
    Road reward function
    """
    def _evaluate_road_heuristic(self, road_action, node_index):
        road_index = np.argmax(road_action)
        road_nodes = EDGES_LIST[road_index]
        target_node = road_nodes[0] if road_nodes[1] == node_index else road_nodes[1]
        possible_nodes = self.__get_other_adjacent_nodes(target_node, node_index)

        nodes_value = self.__evaluate_future_node_values(possible_nodes)
        return nodes_value

    """
    Gives 1 to a 'perfect' node
    """
    def __evaluate_future_node_values(self, possible_nodes):

        def is_node_occupied(node_id):
            adj_tile = NODES_TO_TILES[node_id][0] # 1 of the adj nodes
            nodes = TILES_TO_NODES[adj_tile]
            index = nodes.index(node_id)
            return self._base_obs["nodes_owners"][adj_tile][index].any()

        value = 0
        for node in possible_nodes:
            if not self._is_valid_settlement_placement(node):
                continue
            if is_node_occupied(node):
                value -= 0.5
                continue
            tokens = [
                0 if np.all(tile == 0) else TOKENS[np.argmax(tile)]
                for tile in self._obs["tiles_tokens"][node]
            ]
            for token in tokens:
                if token != 0 :
                    value += DICE_PROBABILITIES[token] / MAX_PROBABILITY
        return value

    def _evaluate_placement(self, settlement_action):
        node_id = np.argmax(settlement_action)
        return self._obs["has_port"][node_id].sum() > 0

    def _evaluate_expected_resource_gain(self, settlement_action):

        def get_adjacent_tiles(node_id):
            return NODES_TO_TILES[node_id]

        def get_adjacent_resources(adjacent_tiles):
            return [
                np.argmax(self._base_obs["resources"][tile])
                for tile in adjacent_tiles
            ]

        def get_adjacent_tokens(adjacent_tiles):
            adjacent_tiles_tokens_ids = [
                np.argmax(self._base_obs["tokens"][tile])
                for tile in adjacent_tiles
            ]
            return [TOKENS[i] for i in adjacent_tiles_tokens_ids]

        def calculate_expected_gains(resources, tokens):
            gains = [0 for _ in range(N_TILE_TYPES)]
            for i in range(len(resources)):
                resource = resources[i]
                token = tokens[i]
                expected_gain = DICE_PROBABILITIES[token] * NUM_ROLLS
                gains[resource] += expected_gain
            gains[-1] = 0  # Desert
            return gains

        def save_settlement_gains(gains):
            player = self._turn_order[self._turn_index]
            self._settlement_gains[player, 0 if self._turn_index <= 3 else 1] = gains

        def normalize_gain_score(gains):
            sum_gain = sum(gains)
            return sum_gain / BEST_EXPECTED_GAIN

        node_id = np.argmax(settlement_action)
        adjacent_tiles = get_adjacent_tiles(node_id)
        resources = get_adjacent_resources(adjacent_tiles)
        tokens = get_adjacent_tokens(adjacent_tiles)
        gains = calculate_expected_gains(resources, tokens)

        save_settlement_gains(gains)
        normalized_gain_score = normalize_gain_score(gains)
        return normalized_gain_score

    def _evaluate_resources_distribution(self, gains):
        player = self._turn_order[self._turn_index]
        gained = gains[player][0] + gains[player][1]

        total = np.sum(gained)
        if total == 0:
            return 0.0

        # Normalize for diversity (entropy)
        probs = gained / total
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        max_entropy = np.log(len(gained))
        diversity_score = entropy / max_entropy

        # Add coverage: fraction of resource types actually gained
        coverage_score = np.count_nonzero(gained) / len(gained)
        coverage_score -= 0.6  # baseline adjustment
        coverage_score *= 2.5  # normalize to [0, 1]

        # Combine them (tunable weights)
        return float(DIVERSITY_SCORE_WEIGHT * diversity_score +
                     COVERAGE_SCORE_WEIGHT * coverage_score)
