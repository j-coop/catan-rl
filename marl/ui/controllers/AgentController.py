import time
from marl.ui.controllers.PlayerController import PlayerController


class AIController(PlayerController):
    def __init__(self, player_name, agent, delay=2.0):
        super().__init__(player_name)
        self.agent = agent
        self.delay = delay

    def request_action(self, game, action_space):
        # Optional thinking delay
        time.sleep(self.delay)

        obs = game.get_observation_for_player(self.player_name)
        mask = action_space.get_action_mask(game.get_player(self.player_name))

        action = self.agent.predict(obs, mask)
        game.apply_action(self.player_name, action)
