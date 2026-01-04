from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from marl.model.ActionLogEntry import ActionLogEntry


class GameManager(QObject):
    log_updated = pyqtSignal()
    ui_updated = pyqtSignal()  # emit whenever UI should refresh

    def __init__(self, game, controllers, action_space, config, ai_delay_ms=1500):
        super().__init__()
        self.game = game
        self.config = config
        self.controllers = controllers
        self.action_space = action_space
        self.ai_delay_ms = ai_delay_ms

        # Set after creation in CatanWindow
        self.action_panel = None
        self.board = None
        self.info_panel = None  # set this too (recommended)

        self.action_logs: list[ActionLogEntry] = []

        self._ai_step_scheduled = False

    def log_action(self, player_name: str, player_color: str, text: str):
        self.action_logs.append(ActionLogEntry(
            player_name=player_name,
            player_color=player_color,
            text=text,
        ))
        self.log_updated.emit()

    def refresh_ui(self):
        """Central place to refresh UI after any action/turn."""
        if self.info_panel:
            self.info_panel.update_all()
        if self.action_panel:
            self.action_panel.update_buttons()
        if self.board:
            self.board.update_roll_display(is_agent=True)
            self.board.update_robber()

        self.ui_updated.emit()

    def on_turn_changed(self):
        """Call this after end_turn, and also once after window shows to start AI."""
        self.refresh_ui()
        self._schedule_ai_if_needed()

    def _schedule_ai_if_needed(self):
        if self._ai_step_scheduled:
            return

        player_name = self.game.current_player.name
        controller = self.controllers[player_name]

        # human - do nothing
        if getattr(controller, "is_human", False):
            return

        # schedule exactly one step
        self._ai_step_scheduled = True
        QTimer.singleShot(self.ai_delay_ms, self._run_ai_step)

    def _run_ai_step(self):
        self._ai_step_scheduled = False

        player_name = self.game.current_player.name
        controller = self.controllers[player_name]

        # One AI action (controller must NOT call on_turn_changed)
        controller.request_action(self.game, self.action_space, self)

        # After action, refresh UI and if still AI, schedule again
        self.refresh_ui()
        self._schedule_ai_if_needed()
