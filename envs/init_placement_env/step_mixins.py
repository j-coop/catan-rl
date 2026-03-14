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

    def __build_road(self, edge_id, player_id):
        edge_coords = EDGES_LIST[edge_id]
        for tile_id in range(N_TILES):
            adj_nodes = TILES_TO_NODES[tile_id]
            for i in range(len(adj_nodes)):
                if (adj_nodes[i], adj_nodes[(i + 1) % 6]) == edge_coords:
                    self._base_obs["edges_owners"][tile_id][i][player_id] = 1

    def _make_settlement_action(self, player, node_id):
        self.__build_settlement(node_id, player)
        self._update_settlement_placement_mask(node_id)
        self.update_road_placement_mask(node_id, player)
        self.last_settlement_node_index = node_id

    def _make_road_action(self, player, road_action):
        self.__build_road(road_action, player)

    def __get_other_adjacent_nodes(self, node, known_node):
        possible_nodes = NODES_TO_NODES[node].copy()
        if known_node in possible_nodes:
            possible_nodes.remove(known_node)
        return possible_nodes

    """
    Road reward function
    """
    def _evaluate_road_heuristic(self, road_index):
        road_nodes = EDGES_LIST[road_index]
        node_built = self.last_settlement_node_index
        target_node = road_nodes[0] if road_nodes[1] == node_built else road_nodes[1]
        possible_nodes = self.__get_other_adjacent_nodes(target_node, node_built)

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

    def _evaluate_placement(self, node_id):
        return self._obs["has_port"][node_id].sum() > 0

    def _evaluate_expected_resource_gain(self, node_id):

        def get_adjacent_tiles(node_id):
            return NODES_TO_TILES.get(node_id, [])

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
            player = self.turn_order[self.turn_index]
            self._settlement_gains[player, 0 if self.turn_index <= 3 else 1] = gains

        def normalize_gain_score(gains):
            sum_gain = sum(gains)
            return sum_gain / BEST_EXPECTED_GAIN

        adjacent_tiles = get_adjacent_tiles(node_id)
        resources = get_adjacent_resources(adjacent_tiles)
        tokens = get_adjacent_tokens(adjacent_tiles)
        gains = calculate_expected_gains(resources, tokens)

        save_settlement_gains(gains)
        normalized_gain_score = normalize_gain_score(gains)
        return normalized_gain_score

    def _evaluate_resources_distribution(self, gains):
        """
        Evaluate quality of the resource distribution for the player's
        two initial settlements using resource importance weights.

        Rewards:
        - covering many different valuable resource types
        - balanced production among them

        Output range approximately: [0, 1]
        """

        player = self.turn_order[(self.turn_index - 1) % len(self.turn_order)]
        gained = gains[player][0] + gains[player][1]

        weights = np.array(TILE_WEIGHTS)
        weighted_gained = gained * weights

        total = np.sum(weighted_gained)
        if total == 0:
            return 0.0

        # ---------- weighted diversity (entropy) ----------
        probs = weighted_gained / total
        entropy = -np.sum(probs * np.log(probs + 1e-9))

        # max entropy only over non-zero weight resources
        valid_resources = weights > 0
        max_entropy = np.log(np.sum(valid_resources))

        diversity_score = entropy / max_entropy

        # ---------- weighted coverage ----------
        # count resources that are both present and useful
        resources_present = np.sum((gained > 0) & valid_resources)
        total_resources = np.sum(valid_resources)

        coverage_score = (resources_present / total_resources) ** 2

        # ---------- combine scores ----------
        diversity_weight = 0.35
        coverage_weight = 0.65

        score = (
            diversity_weight * diversity_score +
            coverage_weight * coverage_score
        )

        return float(score)
