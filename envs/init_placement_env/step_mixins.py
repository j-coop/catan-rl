import random
from math import floor

import numpy as np

from params.catan_constants import TOKENS, DICE_PROBABILITIES, MAX_PROBABILITY, NUM_ROLLS, N_RESOURCE_TYPES, \
    BEST_EXPECTED_GAIN
from params.edges_list import EDGES_LIST
from params.nodes2nodes_adjacency_map import NODES_TO_NODES


class CatanStepMixin:

    def __is_valid_settlement_placement(self, node_id):
        if self._obs["adjacent_nodes"]["is_built"][node_id].any():
            return False  # adjacent node already has a settlement
        return True  # You can add more checks if needed

    def __apply_settlement(self, node_id):
        self._obs["adjacent_nodes"]["is_built"][node_id] = 1
        # self.__obs["adjacent_nodes"]["is_owned"][node_id, i, agent_id] = 1

    def __is_valid_road_placement(self, edge_id):
        # Optional: Add checks for adjacency to a settlement
        return self._obs["edges"]["is_built"][:, edge_id].sum() == 0

    def __apply_road(self, edge_id):
        self._obs["edges"]["is_built"][:, edge_id] = 1
        self._obs["edges"]["is_owned"][:, edge_id] = 1

    def __check_if_placement_done(self):
        # Return True if all agents have finished their initial placements
        # Could track self.__num_settlements or self.__placements_done
        return False
    
    def __verify_action(self, action, settlement_action, road_action):
        assert self.action_space.contains(action), "Invalid action format"
        assert (settlement_action.sum() + road_action.sum()) == 1, \
            "Exactly one action should be performed per step"


    # Both actions should just update observation space

    def __make_settlement_action(self, player, settlement_action):
        node_id = np.argmax(settlement_action)
        if not self.__is_valid_settlement_placement(node_id):
            # zakladam ze sie zamaskuje podanie niemozliwej akcji do step
            reward = -1.0
            terminated = True
            truncated = False
            return self._obs, reward, terminated, truncated, {}

        self.__apply_settlement(node_id)
        self.__update_obs_after_settlement(node_id, player)
        reward = 1.0  # Or 0.0 if using sparse reward
        terminated = self.__check_if_placement_done()
        truncated = False
        return self._obs, reward, terminated, truncated, {}

    def __make_road_action(self, player, road_action):
        edge_id = np.argmax(road_action)
        if not self.__is_valid_road_placement(edge_id):
            # zakladam ze sie zamaskuje podanie niemozliwej akcji do step
            reward = -1.0
            terminated = True
            truncated = False
            return self._obs, reward, terminated, truncated, {}

        self.__apply_road(edge_id)
        self.__update_obs_after_road(edge_id, player)
        reward = 1.0  # Or 0.0
        terminated = self.__check_if_placement_done()
        truncated = False
        return self._obs, reward, terminated, truncated, {}

    def __is_placing_1_settlement(self, settlement_action):
        return settlement_action.sum() == 1
    
    def __is_placing_1_road(self, road_action):
        return road_action.sum() == 1

    def __update_obs_after_settlement(self, node_id, player):
        pass

    def __update_obs_after_road(self, edge_id, player):
        # Edge defining nodes
        nodes = EDGES_LIST[edge_id]
        for node in nodes:
            adjacent_nodes = NODES_TO_NODES[node]
            # Set road as built for both nodes
            self._obs["edges"]["is_built"][node, adjacent_nodes.index(node)] = 1
            # Set ownership maybe baby

    def __get_other_adjacent_nodes(self, node, known_node):
        possible_nodes = NODES_TO_NODES[node]
        possible_nodes.remove(known_node)
        return possible_nodes

    """
    Road reward function
    """
    def __evaluate_road_heuristic(self, road_action, node_index):
        road_index = np.argmax(road_action)
        road_nodes = EDGES_LIST[road_index]
        target_node = road_nodes[0] if road_nodes[1] == node_index else road_nodes[1]
        possible_nodes = self.__get_other_adjacent_nodes(target_node, node_index)

        # Heuristic 1: Estimation of two possible settlements road can lead to (0 if already occupied or impossible)
        potential_value = self.__estimate_future_node_values(possible_nodes)

        # Heuristic 2: Is it directed towards a port?
        is_toward_port = self.__check_if_toward_port(target_node, possible_nodes)

        return potential_value * 0.75 + is_toward_port * 0.25

    """
    Gives 1 to 'perfect' node. Returns sum of values
    """
    def __estimate_future_node_values(self, possible_nodes):
        value = 0
        for node in possible_nodes:
            if not self.__is_valid_settlement_placement(node):
                continue
            tokens = [TOKENS[np.argmax(tile)] for tile in self._obs["tiles"]["tokens"][node]]
            norm_prob = [DICE_PROBABILITIES[token] / MAX_PROBABILITY for token in tokens]
            value += np.mean(norm_prob)
        return value

    def __has_port(self, node):
        return self._obs["has_port"][node].sum() > 0

    """
    Returns points for port chances
    1 - for close port (one more edge)
    0.4 - for far port (two more edges)
    0 - no found port chances
    Assumption: ports are of equal value at the beginning
    """
    def __check_if_toward_port(self, base_node, possible_nodes):
        value = 0
        for node in possible_nodes:
            # Check if road can be placed
            edge_index = EDGES_LIST.index((min(base_node, node), max(base_node, node)))
            if not self.__is_valid_road_placement(edge_index):
                continue
            # Check for possible port
            if self.__has_port(node) and self.__is_valid_settlement_placement(node):
                # Close port - full point
                value += 1
                break
            # Check further port
            further_nodes = self.__get_other_adjacent_nodes(node, base_node)
            for further_node in further_nodes:
                edge_index = EDGES_LIST.index((min(node, further_node), max(node, further_node)))
                if not self.__is_valid_road_placement(edge_index):
                    continue
                # Check for possible port
                if self.__has_port(node) and self.__is_valid_settlement_placement(node):
                    # Far port - lower value
                    value += 0.4
                    break
        return value

    """
    Returns normalized score for gained resources
    """
    def __simulate_dice_rolls(self, settlement_action):
        node_id = np.argmax(settlement_action)
        adjacent_tiles_resources = [np.argmax(tile) for tile in self._obs["tiles"]["resources"][node_id]]
        adjacent_tiles_tokens_ids = [np.argmax(tile) for tile in self._obs["tiles"]["tokens"][node_id]]
        adjacent_tiles_tokens = [TOKENS[i] for i in adjacent_tiles_tokens_ids]
        gains = [0 for _ in range(N_RESOURCE_TYPES)]
        for _ in range(NUM_ROLLS):
            roll = random.randint(1,6) + random.randint(1, 6)
            if roll in adjacent_tiles_tokens:
                for i in range(len(adjacent_tiles_tokens)):
                    gains[adjacent_tiles_resources[i]] += 1
        # Get player, save simulated gain
        player = self._turn_order[self._turn_index]
        self._settlement_gains[player, floor((self._turn_index + 1) / 4)] = gains
        # Award reward
        sum_gain = sum(gains)
        normalized_gain_score = sum_gain / BEST_EXPECTED_GAIN
        return normalized_gain_score

    """
    Returns reward for well distributed resources
    """
    def __evaluate_final_resources(self, gains):
        player = self._turn_order[self._turn_index]
        gained = gains[player][0] + gains[player][1]

        total = np.sum(gained)
        if total == 0:
            return 0
        probs = gained / total  # Normalize to probability distribution
        entropy = -np.sum(probs * np.log(probs + 1e-9))  # Jest jak prosiles xd

        max_entropy = np.log(len(gained))  # Max entropy for uniform distribution
        score = entropy / max_entropy
        return float(score)
