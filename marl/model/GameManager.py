
class GameManager:
    def __init__(self, game, controllers, action_space, config):
        self.game = game
        self.config = config
        self.controllers = controllers
        self.action_space = action_space

    def on_turn_changed(self):
        controller = self.controllers[self.game.current_player.name]
        controller.request_action(self.game, self.action_space)
