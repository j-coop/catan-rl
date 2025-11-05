from params.nodes2nodes_adjacency_map import NODES_TO_NODES


class CatanValidationMixin:

    def _is_placing_settlement(self, settlement_action):
        return settlement_action.sum() == 1
    
    def _is_placing_road(self, road_action):
        return road_action.sum() == 1
    
    def _is_valid_settlement_placement(self, node_id):
        if self._obs["adj_is_built"][node_id].any():
            for adj_node_id in NODES_TO_NODES[node_id]:
                if self._obs["adj_is_built"][adj_node_id].any():
                    return False 
        else: return False  # adjacent node already has a settlement
        return True

    def _is_valid_road_placement(self, edge_id):
        # TODO Fix this func
        return True
        # return self._obs["edges_is_built"][:, edge_id].sum() == 0
    
    def _verify_action(self, action, settlement_action, road_action):
        assert self.action_space.contains(action), "Invalid action format"
        assert (settlement_action.sum() + road_action.sum()) == 1, \
            "Exactly one action should be performed per step"
