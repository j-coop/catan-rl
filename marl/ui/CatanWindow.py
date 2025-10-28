from PyQt6.QtWidgets import QWidget, QHBoxLayout

from marl.model.CatanGame import CatanGame
from marl.ui.ActionPanel import ActionPanel
from marl.ui.PlayerInfoPanel import PlayerInfoPanel
from marl.ui.board_view import BoardView


class CatanWindow(QWidget):
    """Main game window combining board + info panels."""

    def __init__(self, game: CatanGame):
        super().__init__()
        self.game = game
        self.setWindowTitle("Settlers of Catan RL")

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.info_panel = PlayerInfoPanel(game)
        self.board = BoardView(hex_radius=55)
        self.action_panel = ActionPanel(game)

        layout.addWidget(self.info_panel)
        layout.addWidget(self.board, 1)
        layout.addWidget(self.action_panel)

        self.setMinimumSize(1200, 800)
