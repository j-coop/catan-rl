from marl.ui.controllers.PlayerController import PlayerController


class HumanController(PlayerController):
    def request_action(self, game, action_space, game_manager):
        # Human actions come via UI, so nothing to do here
        pass
