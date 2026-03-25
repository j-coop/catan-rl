import numpy as np
from tianshou.data import Batch
from params.catan_constants import BANK_TRADE_PAIRS, RESOURCE_TYPES, DICE_PROBABILITIES
from params.nodes2tiles_adjacency_map import NODES_TO_TILES

class HeuristicCatanPolicy:
    """
    A rule-based Tianshou policy for Catan.
    Levels:
    1 - Random valid action
    2 - Prioritizes building (City -> Settlement -> Dev Card -> Road -> End Turn). No bank trades to prevent loops.
    3 - Similar to 2, but makes randomized bank trades and has a higher preference for dev cards.
    """
    def __init__(self, level=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.level = level

    def forward(self, batch: Batch, state=None, **kwargs):
        masks = batch.obs["action_mask"]
        obs_vecs = batch.obs["observation"]
        actions = []

        for i in range(len(masks)):
            mask = masks[i]
            obs_vec = obs_vecs[i]
            valid_actions = np.where(mask == 1)[0]
            
            if len(valid_actions) == 0:
                actions.append(230) # Fallback to end turn
                continue

            if self.level == 1:
                actions.append(np.random.choice(valid_actions))
            elif self.level >= 2:
                action = self._choose_heuristic_action(mask, valid_actions, self.level, obs_vec)
                actions.append(action)

        return Batch(act=np.array(actions), state=state)

    def learn(self, batch, **kwargs):
        # Heuristic bots do not learn
        return {}

    def _parse_obs(self, obs_vec):
        # BOARD_SPACE_SIZE = 1290
        # SELF_SPACE_SIZE = 23
        # Tile feats: 19 * 12
        # Tile: [0:6] res, [6] num, [7] robber, [8] self_prod, [9] opp_prod, [10] self_has, [11] opp_has
        
        tile_feats = obs_vec[0:228].reshape(19, 12)
        self_feats = obs_vec[1290:1313]
        
        res_counts = self_feats[0:5] # wood, brick, sheep, wheat, ore
        total_res = np.sum(res_counts) * 19 # normalized by MAX_RESOURCE_COUNT
        
        # Production by resource type (0-4)
        prod = np.zeros(5)
        for i in range(19):
            res_onehot = tile_feats[i, 0:5]
            if res_onehot.any():
                res_idx = np.where(res_onehot == 1)[0][0]
                prod[res_idx] += tile_feats[i, 8] # self_prod
                
        return {
            "tile_feats": tile_feats,
            "res_counts": res_counts,
            "total_res": total_res,
            "prod": prod
        }

    def _score_node(self, node_idx, tile_feats, obs_vec):
        score = 0.0
        adjacent_tiles = NODES_TO_TILES.get(node_idx, [])
        for tid in adjacent_tiles:
            tile = tile_feats[tid]
            # tile: [0:6] res, [6] token/12, [7] robber, [8] self_prod, [9] opp_prod
            if tile[0:5].any(): # Not desert
                num = int(round(tile[6] * 12.0))
                prob = DICE_PROBABILITIES.get(num, 0.0)
                score += prob
            
        return score

    def _choose_heuristic_action(self, mask, valid_actions, level, obs_vec):
        info = self._parse_obs(obs_vec)
        
        # 1. Mandatory actions: move_robber
        if mask[186:205].any() and not mask[230]:
            tile_scores = []
            for i in range(19):
                # Penalty for blocking self
                score = -info["tile_feats"][i, 8] * 10
                if info["tile_feats"][i, 10] > 0: # self_has_building
                    score -= 50
                # Reward for blocking opponent productivity
                score += info["tile_feats"][i, 9] * 20
                if info["tile_feats"][i, 11] > 0: # opp_has_building
                    score += 10
                tile_scores.append(score)
            
            # Mask out invalid tiles (masks[186:205])
            valid_robber_indices = np.where(mask[186:205] == 1)[0]
            best_tile_idx = valid_robber_indices[np.argmax([tile_scores[idx] for idx in valid_robber_indices])]
            return best_tile_idx + 186
        
        # 2. Mandatory actions: choose_resource (Year of Plenty / Monopoly)
        if mask[225:230].any() and not mask[230]:
            # Choose resource with lowest production or currently missing one
            valid_res_indices = np.where(mask[225:230] == 1)[0]
            best_res_idx = valid_res_indices[np.argmin(info["prod"][valid_res_indices])]
            return best_res_idx + 225

        # Normal phase priorities
        # 3. Settlement (Higher priority than City as requested)
        valid_settlements = np.where(mask[0:54] == 1)[0]
        if len(valid_settlements) > 0:
            if level == 3:
                scores = [self._score_node(idx, info["tile_feats"], obs_vec) for idx in valid_settlements]
                return valid_settlements[np.argmax(scores)]
            return np.random.choice(valid_settlements)

        # 4. City
        valid_cities = np.where(mask[54:108] == 1)[0]
        if len(valid_cities) > 0:
            if level == 3:
                # Same production scoring for existing settlements
                scores = [self._score_node(idx, info["tile_feats"], obs_vec) for idx in valid_cities]
                return valid_cities[np.argmax(scores)] + 54
            return np.random.choice(valid_cities) + 54

        # 5. Build Road (Prioritize if no settlements possible)
        valid_roads = np.where(mask[108:180] == 1)[0]
        if len(valid_roads) > 0:
            # If we don't have settlements possible, prioritize roads
            if len(valid_settlements) == 0 or np.random.rand() < 0.5:
                return np.random.choice(valid_roads) + 108

        # 6. Trade Bank (Level 3 specific logic)
        if level == 3 and info["total_res"] > 7:
            valid_trades = np.where(mask[205:225] == 1)[0]
            if len(valid_trades) > 0:
                # BANK_TRADE_PAIRS = [("wood", "brick"), ...]
                # Trade only for resources it doesn't produce
                best_trade_idx = -1
                for trade_idx in valid_trades:
                    # recv_idx is trade_idx % 4 (relative) or more accurately from BANK_TRADE_PAIRS
                    give_res, take_res = BANK_TRADE_PAIRS[trade_idx]
                    take_idx = RESOURCE_TYPES.index(take_res)
                    if info["prod"][take_idx] == 0:
                        best_trade_idx = trade_idx
                        break
                if best_trade_idx != -1:
                    return best_trade_idx + 205

        # 7. Play Dev Card
        valid_play_dev = np.where(mask[181:186] == 1)[0]
        if len(valid_play_dev) > 0:
            if np.random.rand() < 0.5:
                return np.random.choice(valid_play_dev) + 181

        # 8. Buy Dev Card
        if mask[180] == 1:
            if np.random.rand() < 0.2:
                return 180

        # 9. End Turn
        if mask[230] == 1:
            return 230
            
        return np.random.choice(valid_actions)
