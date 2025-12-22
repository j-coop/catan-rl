from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton

from marl.env.ActionSpace import ActionSpace
from marl.model.CatanGame import CatanGame
from marl.ui.ChoiceGridDialog import ChoiceGridDialog
from marl.ui.ChoiceOption import ChoiceOption
from marl.ui.EnvMock import EnvMock
from marl.ui.board_view import BoardView
from marl.ui.PlayerInfoPanel import PlayerInfoPanel
from params.catan_constants import BANK_TRADE_PAIRS


class ActionHandler:

    def on_build_settlement(self):
        player = self.game.current_player.name
        board: BoardView = self.parent().findChild(BoardView)
        info_panel: PlayerInfoPanel = self.parent().findChild(PlayerInfoPanel)

        def callback(node_index):
            self.game.build_settlement(player, node_index)
            board.build_settlement_ui(node_index)
            info_panel._update_after_game_change()
        board.expect_node_selection(callback)

    def on_build_road(self):
        player = self.game.current_player.name
        board: BoardView = self.parent().findChild(BoardView)
        info_panel: PlayerInfoPanel = self.parent().findChild(PlayerInfoPanel)

        def callback(edge_index):
            self.game.build_road(player, edge_index)
            board.build_road_ui(edge_index)
            info_panel._update_after_game_change()
        board.expect_edge_selection(callback)

    def on_build_city(self):
        player = self.game.current_player.name
        board: BoardView = self.parent().findChild(BoardView)
        info_panel: PlayerInfoPanel = self.parent().findChild(PlayerInfoPanel)

        def callback(node_index):
            self.game.build_city(player, node_index)
            board.upgrade_city_ui(node_index)
            info_panel._update_after_game_change()
        board.expect_node_selection(callback)

    def on_buy_dev_card(self):
        player = self.game.current_player.name
        info_panel: PlayerInfoPanel = self.parent().findChild(PlayerInfoPanel)
        self.game.buy_dev_card(player)
        info_panel._update_after_game_change()

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
        self.game.end_turn(is_ui_action=True)
        self.info_panel.refresh()
        self.board_view.update_roll_display()

    def show_dev_card_dialog(self):
        options = [
            ChoiceOption(
                text="🗡 Knight",
                enabled=True,
                callback=lambda: self.game.play_knight(),
            ),
            ChoiceOption(
                text="🏆 Victory Point",
                enabled=True,
                callback=lambda: self.game.play_victory_point(),
            ),
            ChoiceOption(
                text="🛣 Road Building",
                enabled=True,
                callback=lambda: self.game.play_road_building(),
            ),
            ChoiceOption(
                text="📦 Monopoly",
                enabled=True,
                callback=lambda: self.game.play_monopoly(),
            ),
            ChoiceOption(
                text="🌾 Year of Plenty",
                enabled=True,
                callback=lambda: self.game.play_year_of_plenty(),
            ),
        ]

        dlg = ChoiceGridDialog(
            title="Play Development Card",
            options=options,
            columns=2,
            parent=self,
        )
        dlg.exec()

    def show_bank_trade_dialog(self):
        resources = ["🪵", "🧱", "🐑", "🌾", "🪨"]
        resources_names = ["wood", "brick", "sheep", "wheat", "ore"]
        options = []

        for give in resources:
            for receive in resources:
                if give == receive:
                    continue

                give_index = resources.index(give)
                receive_index = resources.index(receive)

                trade_index = BANK_TRADE_PAIRS.index((resources_names[give_index], resources_names[receive_index]))
                enabled = self.action_masks.is_action_enabled(self.game.current_player, "trade_bank", trade_index)

                def make_trade_callback(g, r):
                    def handler():
                        self.game.trade_with_bank(self.game.current_player.name, g, r)
                        self.info_panel.refresh()

                    return handler

                options.append(
                    ChoiceOption(
                        text=f"{give} → {receive}",
                        enabled=enabled,
                        callback=make_trade_callback(
                            resources_names[give_index],
                            resources_names[receive_index],
                        )
                    )
                )

        dlg = ChoiceGridDialog(
            title="Trade with Bank (4:1)",
            options=options,
            columns=4,
            parent=self,
        )
        dlg.exec()


class ActionPanel(QWidget, ActionHandler):
    """Right-side control buttons."""

    def __init__(self, game: CatanGame, board_view: BoardView, info_panel: PlayerInfoPanel):
        super().__init__()
        self.game = game
        self.board_view = board_view
        self.info_panel = info_panel

        # Dummy env with game object for ActionSpace
        env = EnvMock(self.game)
        self.action_masks = ActionSpace(env)

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
            "Play Dev Card": self.show_dev_card_dialog,
            "Trade": self.show_bank_trade_dialog,
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
