import numpy as np
from tianshou.policy import BasePolicy
from tianshou.data import Batch

class HeuristicCatanPolicy(BasePolicy):
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
        actions = []

        for i in range(len(masks)):
            mask = masks[i]
            valid_actions = np.where(mask == 1)[0]
            
            if len(valid_actions) == 0:
                actions.append(230) # Fallback to end turn
                continue

            if self.level == 1:
                actions.append(np.random.choice(valid_actions))
            elif self.level >= 2:
                action = self._choose_heuristic_action(mask, valid_actions, self.level)
                actions.append(action)

        return Batch(act=np.array(actions), state=state)

    def learn(self, batch, **kwargs):
        # Heuristic bots do not learn
        return {}

    def _choose_heuristic_action(self, mask, valid_actions, level):
        # Action space mapping:
        # 0-53: build_settlement
        # 54-107: build_city
        # 108-179: build_road
        # 180: buy dev card
        # 181-185: play dev card
        # 186-204: move robber
        # 205-224: trade bank
        # 225-229: choose resource
        # 230: end turn

        # Check for forced actions (robber or choose resource)
        # If any of these are the ONLY valid actions (excluding end_turn which isn't valid during robber phase)
        if mask[186:205].any() and not mask[230]:
            return np.random.choice(np.where(mask[186:205] == 1)[0]) + 186
        
        if mask[225:230].any() and not mask[230]:
            return np.random.choice(np.where(mask[225:230] == 1)[0]) + 225

        # Normal phase priorities
        # 1. City
        valid_cities = np.where(mask[54:108] == 1)[0]
        if len(valid_cities) > 0:
            return np.random.choice(valid_cities) + 54

        # 2. Settlement
        valid_settlements = np.where(mask[0:54] == 1)[0]
        if len(valid_settlements) > 0:
            return np.random.choice(valid_settlements)

        # 3. Buy Dev Card
        if mask[180] == 1:
            # Level 3 buys cards more aggressively. Level 2 has 50% chance.
            if level == 3 or np.random.rand() < 0.5:
                return 180

        # 4. Play Dev Card
        valid_play_dev = np.where(mask[181:186] == 1)[0]
        if len(valid_play_dev) > 0:
            # Play a dev card with 30% chance to avoid dumping everything at once
            if np.random.rand() < 0.3:
                return np.random.choice(valid_play_dev) + 181

        # 5. Trade Bank (Level 3 only, 20% chance to avoid infinite loops)
        if level == 3:
            valid_trades = np.where(mask[205:225] == 1)[0]
            if len(valid_trades) > 0 and np.random.rand() < 0.2:
                return np.random.choice(valid_trades) + 205

        # 6. Build Road
        valid_roads = np.where(mask[108:180] == 1)[0]
        if len(valid_roads) > 0:
            # 50% chance to build a road if possible, else save resources
            if np.random.rand() < 0.5:
                return np.random.choice(valid_roads) + 108

        # 7. End Turn
        if mask[230] == 1:
            return 230
            
        # Fallback if somehow nothing above hit but we have valid actions
        return np.random.choice(valid_actions)
