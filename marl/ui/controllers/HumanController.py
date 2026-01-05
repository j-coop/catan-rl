from marl.ui.controllers.PlayerController import PlayerController


class HumanController(PlayerController):
    def __init__(self, player_name: str):
        super().__init__(player_name)
        self.is_human = True

    def request_action(self, game, action_space, game_manager):
        # Human actions come via UI, so nothing to do here
        pass
