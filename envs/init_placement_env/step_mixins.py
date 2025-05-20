import numpy as np


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
            reward = -1.0
            terminated = True
            truncated = False
            return self._obs, reward, terminated, truncated, {}

        self.__apply_settlement(node_id)
        self.__update_obs_after_settlement(node_id, player)
        reward = 1.0  # Or 0.0 if using sparse reward
        terminated = self.__check_if_placement_done()
        truncated = False
        return self.__obs, reward, terminated, truncated, {}

    def __make_road_action(self, player, road_action):
        edge_id = np.argmax(road_action)
        if not self.__is_valid_road_placement(edge_id):
            reward = -1.0
            terminated = True
            truncated = False
            return self.__obs, reward, terminated, truncated, {}

        self.__apply_road(edge_id)
        self.__update_obs_after_road(edge_id, player)
        reward = 1.0  # Or 0.0
        terminated = self.__check_if_placement_done()
        truncated = False
        return self.__obs, reward, terminated, truncated, {}

    def __is_placing_1_settlement(self, settlement_action):
        return settlement_action.sum() == 1
    
    def __is_placing_1_road(self, road_action):
        return road_action.sum() == 1

    def __update_obs_after_settlement(self, node_id, player):
        pass

    def __update_obs_after_road(self, edge_id, player):
        pass
