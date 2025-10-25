from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QLabel, QSizePolicy, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class MainWindow(QMainWindow):
    def __init__(self, controller=None):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Settlers of Catan – RL Testing Interface")
        self.setMinimumSize(1200, 800)

        # Central container
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # === Left Panel ===
        self.left_panel = self.create_side_panel(title="Player Info / Log")
        main_layout.addWidget(self.left_panel, 2)

        # === Center Board ===
        self.board_view = self.create_board_placeholder()
        main_layout.addWidget(self.board_view, 5)

        # === Right Panel ===
        self.right_panel = self.create_action_panel()
        main_layout.addWidget(self.right_panel, 2)

    # -------------------------
    # Helper UI builders
    # -------------------------
    def create_side_panel(self, title: str) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout()
        panel.setLayout(layout)

        label = QLabel(title)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(label)

        # Placeholder content
        info = QLabel("Game messages and player stats\nwill appear here.")
        info.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(info)

        layout.addStretch()
        return panel

    def create_board_placeholder(self) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.Box)
        layout = QVBoxLayout()
        frame.setLayout(layout)

        label = QLabel("BOARD AREA")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        layout.addWidget(label)
        return frame

    def create_action_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout()
        panel.setLayout(layout)

        title = QLabel("Actions")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # --- 3x2 grid for buttons ---
        grid = QGridLayout()
        grid.setSpacing(10)

        button_style = """
            QPushButton {
                background-color: #1976d2;
                color: white;
                border-radius: 6px;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
        """

        buttons = [
            ("Build Settlement", self.on_build_settlement),
            ("Upgrade to City", self.on_upgrade_city),
            ("Build Road", self.on_build_road),
            ("Buy Dev Card", self.on_buy_dev_card),
            ("Play Dev Card", self.on_play_dev_card),
            ("Trade", self.on_trade),
        ]

        for i, (text, handler) in enumerate(buttons):
            btn = QPushButton(text)
            btn.setStyleSheet(button_style)
            btn.clicked.connect(handler)
            grid.addWidget(btn, i // 2, i % 2)

        layout.addLayout(grid)

        # --- End Turn button (full width) ---
        end_turn_btn = QPushButton("End Turn")
        end_turn_btn.setStyleSheet("""
            QPushButton {
                background-color: #388e3c;
                color: white;
                border-radius: 6px;
                font-size: 16px;
                padding: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2e7d32;
            }
            QPushButton:pressed {
                background-color: #1b5e20;
            }
        """)
        end_turn_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        end_turn_btn.clicked.connect(self.on_end_turn)
        layout.addSpacing(20)
        layout.addWidget(end_turn_btn)

        layout.addStretch()
        return panel

    # -------------------------
    # Action handlers (connect to controller)
    # -------------------------
    def on_build_settlement(self):
        print("Build Settlement clicked")

    def on_upgrade_city(self):
        print("Upgrade to City clicked")

    def on_build_road(self):
        print("Build Road clicked")

    def on_buy_dev_card(self):
        print("Buy Dev Card clicked")

    def on_play_dev_card(self):
        print("Play Dev Card clicked")

    def on_trade(self):
        print("Trade clicked")

    def on_end_turn(self):
        print("End Turn clicked")
