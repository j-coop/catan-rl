import numpy as np
from tianshou.data import Batch

from marl.ui.controllers.AgentController import AgentController
from marl.env.tianshou.heuristic_bot import HeuristicCatanPolicy

class BotController(AgentController):
    """
    Extends AgentController to use the rule-based HeuristicCatanPolicy 
    instead of the PyTorch neural network actor.
    """
    def __init__(self, player_name, agent, level=1, delay=2.0):
        super().__init__(player_name, agent, delay)
        self.bot_policy = HeuristicCatanPolicy(level=level)

    def _ensure_actor(self, action_space):
        # We don't need a torch actor for the heuristic bot
        pass

    def _infer_action(self, action_space, mask):
        valid_indices = [i for i, v in enumerate(mask) if v]
        if not valid_indices:
            print("Bot has no valid actions. Falling back to end turn.")
            return 230

        obs_vec = action_space.env.get_observation(self.player_name)
        mask_arr = np.array(mask, dtype=int)
        
        # Construct the Batch object that the tianshou policy expects
        batch = Batch(
            obs=Batch(
                observation=np.array([obs_vec]),
                action_mask=np.array([mask_arr])
            ),
            info={}
        )
        
        result_batch = self.bot_policy.forward(batch)
        action = result_batch.act[0]
        return int(action)
