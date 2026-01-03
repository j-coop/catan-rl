from PyQt6.QtCore import QObject, pyqtSignal

from marl.model.ActionLogEntry import ActionLogEntry


class GameManager(QObject):
    log_updated = pyqtSignal()

    def __init__(self, game, controllers, action_space, config):
        super().__init__()
        self.game = game
        self.config = config
        self.controllers = controllers
        self.action_space = action_space

        # Set after creation in CatanWindow
        self.action_panel = None
        self.board = None

        self.action_logs: list[ActionLogEntry] = []

    def log_action(self, player_name: str, player_color: str, text: str):

        entry = ActionLogEntry(
            player_name=player_name,
            player_color=player_color,
            text=text,
        )

        self.action_logs.append(entry)
        self.log_updated.emit()

    def on_turn_changed(self):
        controller = self.controllers[self.game.current_player.name]
        controller.request_action(self.game, self.action_space, self)
