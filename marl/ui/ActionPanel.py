from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton

from marl.model.CatanGame import CatanGame
from marl.ui.board_view import BoardView

class ActionHandler:

    def on_build_settlement(self):
        player = self.game.current_player.name
        board: BoardView = self.parent().findChild(BoardView)

        def callback(node_index):
            self.game.build_settlement(player, node_index)

        board.expect_node_selection(callback)

    def on_build_city(self):
        player = self.game.current_player.name
        board: BoardView = self.parent().findChild(BoardView)

        def callback(node_index):
            self.game.build_city(player, node_index)

        board.expect_node_selection(callback)

    def on_build_road(self):
        player = self.game.current_player.name
        board: BoardView = self.parent().findChild(BoardView)

        def callback(edge_index):
            self.game.build_road(player, edge_index)
            # refresh UI if needed

        board.expect_edge_selection(callback)

    def on_buy_dev_card(self):
        player = self.game.current_player.name
        self.game.buy_dev_card(player)

    def on_play_dev_card(self):
        player = self.game.current_player.name
        card_type = self.ask_user_for_dev_card_type()
        if card_type:
            self.game.play_dev_card(player, card_type)

    def on_trade(self):
        player = self.game.current_player.name
        trade_index = self.ask_user_for_trade_option()
        if trade_index is not None:
            self.game.trade_bank(player, trade_index)

    def on_end_turn(self):
        self.game.end_turn()


class ActionPanel(QWidget,
                  ActionHandler):
    """Right-side control buttons."""

    def __init__(self, game: CatanGame):
        super().__init__()
        self.game = game
        self.setFixedWidth(220)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("Actions")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)

        # Map button text to handler method
        self.button_handlers = {
            "Build Settlement": self.on_build_settlement,
            "Upgrade to City": self.on_build_city,
            "Build Road": self.on_build_road,
            "Buy Dev Card": self.on_buy_dev_card,
            "Play Dev Card": self.on_play_dev_card,
            "Trade": self.on_trade,
            "End Turn": self.on_end_turn,
        }

        for operation, handler in self.button_handlers.items():
            btn = QPushButton(operation)
            btn.setFixedHeight(40)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4682B4; color: white;
                    border-radius: 8px; font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #5A9BD6;
                }
            """)
            btn.clicked.connect(handler)
            layout.addWidget(btn)

        layout.addStretch(1)
