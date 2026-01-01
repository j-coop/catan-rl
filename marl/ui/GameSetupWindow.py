from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton
)
from PyQt6.QtGui import QFont, QGuiApplication
from PyQt6.QtCore import Qt

from PyQt6_SwitchControl import SwitchControl


class GameSetupWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Game Setup")
        self.setFixedSize(480, 360)
        self.center_on_screen()

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(18)
        main_layout.setContentsMargins(30, 25, 30, 25)

        # Title
        title = QLabel("Game Setup")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        main_layout.addWidget(title)

        # Player rows
        self.player_rows = []
        players = [
            ("Blue Player", "#2c3aff"),
            ("Purple Player", "#c525c5"),
            ("Yellow Player", "#dbc33a"),
            ("Green Player", "#32a852"),
        ]

        for name, color in players:
            row = QHBoxLayout()
            row.setSpacing(10)

            name_label = QLabel(name)
            name_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            name_label.setFont(QFont("Arial", 11))

            human_label = QLabel("Human")
            human_label.setStyleSheet("color: #666;")

            ai_label = QLabel("AI")
            ai_label.setStyleSheet("color: #666;")

            switch = SwitchControl(
                bg_color="#999999",
                circle_color="#FFFFFF",
                active_color="#4682B4"
            )
            switch.setFixedSize(60, 30)

            row.addWidget(name_label)
            row.addStretch()
            row.addWidget(human_label)
            row.addWidget(switch)
            row.addWidget(ai_label)

            main_layout.addLayout(row)
            self.player_rows.append((name, switch))

        # Start button
        self.start_btn = QPushButton("Start Game")
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setFixedHeight(44)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4682B4;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #5A9BD6;
            }
            QPushButton:pressed {
                background-color: #3b6e99;
            }
        """)
        self.start_btn.clicked.connect(self.start_game)

        main_layout.addSpacing(12)
        main_layout.addWidget(self.start_btn)

    def start_game(self):
        config = {
            name: switch.isChecked()
            for name, switch in self.player_rows
        }
        self.launch_game(config)

    def center_on_screen(self):
        screen = QGuiApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        window_geometry = self.frameGeometry()
        window_geometry.moveCenter(screen_geometry.center())
        self.move(window_geometry.topLeft())
