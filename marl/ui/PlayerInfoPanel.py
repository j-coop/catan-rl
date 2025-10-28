from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame

from marl.model.CatanGame import CatanGame


class PlayerInfoPanel(QWidget):
    """Left-side panel showing player stats."""

    def __init__(self, game: CatanGame):
        super().__init__()
        self.game = game
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.setFixedWidth(180)
        self.refresh()

    def refresh(self):
        # Clear old content
        for i in reversed(range(self.layout.count())):
            w = self.layout.itemAt(i).widget()
            if w:
                w.deleteLater()

        title = QLabel("Players")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.layout.addWidget(title)

        for player in self.game.players:
            block = QFrame()
            block.setFrameShape(QFrame.Shape.StyledPanel)
            block.setStyleSheet(f"background-color: {player.color}; border-radius: 8px;")
            v = QVBoxLayout()

            name_label = QLabel(f"{player.color}")
            name_label.setStyleSheet("color: white; font-weight: bold;")
            v.addWidget(name_label)

            points_label = QLabel(f"Points: {player.victory_points}")
            v.addWidget(points_label)

            res_summary = ", ".join(f"{r[0].upper()}:{r[1]}" for r in player.resources.items())
            resources_label = QLabel(f"Resources: {res_summary}")
            v.addWidget(resources_label)

            block.setLayout(v)
            self.layout.addWidget(block)

        self.layout.addStretch(1)
