from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QWidget, QHBoxLayout

from marl.model.CatanGame import CatanGame
from marl.model.GameManager import GameManager
from marl.ui.ActionPanel import ActionPanel
from marl.ui.PlayerInfoPanel import PlayerInfoPanel
from marl.ui.board_view import BoardView


class CatanWindow(QWidget):
    """Main game window combining board + info panels."""

    def __init__(self, game: CatanGame, game_manager: GameManager):
        super().__init__()
        self.game = game
        self.game_manager = game_manager

        self.setWindowTitle("Settlers of Catan RL")

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.info_panel = PlayerInfoPanel(game, self.game_manager.config)
        self.board = BoardView(info_panel=self.info_panel, hex_radius=65, game=game)
        self.game_manager.board = self.board
        self.action_panel = ActionPanel(game, self.board, self.info_panel, self.game_manager)
        self.board.action_panel = self.action_panel

        layout.addWidget(self.info_panel)
        layout.addWidget(self.board, 1)
        layout.addWidget(self.action_panel)

        self.setMinimumSize(1200, 800)
        self.center_on_screen()

    def center_on_screen(self):
        screen = QGuiApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        window_geometry = self.frameGeometry()
        window_geometry.moveCenter(screen_geometry.center())
        self.move(window_geometry.topLeft())
