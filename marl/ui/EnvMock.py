from marl.env.common import EnvActionHandlerMixin
from marl.env.Rewards import Rewards

class EnvMock(EnvActionHandlerMixin):
    def __init__(self, game):
        self.game = game
        self.reward_object = Rewards(game)
        self.agents = [p.name for p in game.players]
        # We don't need a real ActionSpace here, but handlers might need it for is_end_turn_action
        self.actions = None 

    def get_observation_space_size(self) -> int:
        return 0 # Not used for UI masking
