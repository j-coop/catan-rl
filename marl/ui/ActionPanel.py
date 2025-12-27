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
from params.catan_constants import BANK_TRADE_PAIRS, DEV_CARD_TYPES, N_TILES


class ActionHandler:

    def on_build_settlement(self):
        player = self.game.current_player
        board: BoardView = self.parent().findChild(BoardView)
        info_panel: PlayerInfoPanel = self.parent().findChild(PlayerInfoPanel)

        def callback(node_index):
            if self.action_masks.is_action_enabled(player=player, name="build_settlement", index=node_index):
                self.game.build_settlement(player.name, node_index)
                board.build_settlement_ui(node_index)
                info_panel._update_after_game_change()
                self.update_buttons()
            else:
                print("Invalid node chosen!")
        board.expect_node_selection(callback)

    def on_build_road(self):
        player = self.game.current_player
        board: BoardView = self.parent().findChild(BoardView)
        info_panel: PlayerInfoPanel = self.parent().findChild(PlayerInfoPanel)

        def callback(edge_index):
            if self.action_masks.is_action_enabled(player=player, name="build_road", index=edge_index):
                self.game.build_road(player.name, edge_index)
                board.build_road_ui(edge_index)
                info_panel._update_after_game_change()
                self.update_buttons()  # necessary here after async callback
            else:
                print("Invalid edge chosen!")
        board.expect_edge_selection(callback)

    def on_build_city(self):
        player = self.game.current_player
        board: BoardView = self.parent().findChild(BoardView)
        info_panel: PlayerInfoPanel = self.parent().findChild(PlayerInfoPanel)

        def callback(node_index):
            if self.action_masks.is_action_enabled(player=player, name="build_city", index=node_index):
                self.game.build_city(player.name, node_index)
                board.upgrade_city_ui(node_index)
                info_panel._update_after_game_change()
                self.update_buttons()
            else:
                print("Invalid edge chosen!")
        board.expect_node_selection(callback)

    def on_buy_dev_card(self):
        player = self.game.current_player.name
        info_panel: PlayerInfoPanel = self.parent().findChild(PlayerInfoPanel)
        self.game.buy_dev_card(player)
        info_panel._update_after_game_change()
        self.update_buttons()

    def on_end_turn(self):
        self.game.end_turn(is_ui_action=True)
        self.info_panel.refresh()
        self.board_view.update_roll_display()

    def show_dev_card_dialog(self):
        current_player = self.game.current_player

        def make_callback(card_type: str):
            def callback():
                if card_type == "knight":
                    def callback(hex_index: int):
                        self.game.move_robber(self.game.current_player.name, hex_index)
                        self.board_view.update_robber()
                        self.board_view.clear_hex_selection()
                        self.info_panel._update_after_game_change()
                        self.update_buttons()

                    self.board_view.expect_hex_selection(callback, [i for i in range(0, N_TILES)])

                # Call the unified game handler
                self.game.play_dev_card(current_player.name, card_type)

                self.board_view.update_roll_display()
                self.info_panel.refresh()
                self.update_buttons()

            return callback

        options = [
            ChoiceOption(
                text="🗡 Knight",
                enabled=self.action_masks.is_action_enabled(
                    current_player, "play_dev_card",
                    DEV_CARD_TYPES.index("knight")
                ),
                callback=make_callback("knight"),
            ),
            ChoiceOption(
                text="🏆 Victory Point",
                enabled=self.action_masks.is_action_enabled(
                    current_player, "play_dev_card",
                    DEV_CARD_TYPES.index("victory_point")
                ),
                callback=make_callback("victory_point"),
            ),
            ChoiceOption(
                text="🛣 Road Building",
                enabled=self.action_masks.is_action_enabled(
                    current_player, "play_dev_card",
                    DEV_CARD_TYPES.index("road_building")
                ),
                callback=make_callback("road_building"),
            ),
            ChoiceOption(
                text="📦 Monopoly",
                enabled=self.action_masks.is_action_enabled(
                    current_player, "play_dev_card",
                    DEV_CARD_TYPES.index("monopoly")
                ),
                callback=make_callback("monopoly"),
            ),
            ChoiceOption(
                text="🌾 Year of Plenty",
                enabled=self.action_masks.is_action_enabled(
                    current_player, "play_dev_card",
                    DEV_CARD_TYPES.index("year_of_plenty")
                ),
                callback=make_callback("year_of_plenty"),
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
                        self.update_buttons()

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

        self.buttons = {}  # Store buttons for dynamic updating
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
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #888888;
                    border: 2px solid #999999;
                    font-weight: normal;
                }
            """)

            # Wrap the original handler with update buttons
            def make_handler(h=handler, b=self):
                def wrapped():
                    h()  # call original action
                    b.update_buttons()  # refresh button states

                return wrapped

            btn.clicked.connect(make_handler())
            layout.addWidget(btn)
            self.buttons[operation] = btn

        layout.addStretch(1)
        self.update_buttons()  # Initial update

    def update_buttons(self):
        """Enable/disable buttons according to current player's legal actions."""
        player = self.game.current_player
        mask = self.action_masks.get_action_mask(player)

        # Map button names to ActionSpec names
        mapping = {
            "Build Settlement": "build_settlement",
            "Upgrade to City": "build_city",
            "Build Road": "build_road",
            "Buy Dev Card": "buy_dev_card",
            "Play Dev Card": "play_dev_card",
            "Trade": "trade_bank",
            "End Turn": "end_turn",
        }

        for btn_name, action_name in mapping.items():
            btn = self.buttons[btn_name]
            btn.setEnabled(self.action_masks.is_action_enabled(player=player, name=action_name, mask=mask)
                           and not self.game.game_over)

        self.board_view.update_roll_display()
