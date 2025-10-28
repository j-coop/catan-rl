from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton

from marl.model.CatanGame import CatanGame


class ActionPanel(QWidget):
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

        buttons = [
            "Build Settlement",
            "Upgrade to City",
            "Build Road",
            "Buy Dev Card",
            "Play Dev Card",
            "Trade",
            "End Turn",
        ]
        for text in buttons:
            btn = QPushButton(text)
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
            layout.addWidget(btn)

        layout.addStretch(1)
