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
        self.setFixedWidth(210)
        self.refresh()

    def refresh(self):
        self._clear_old_content()

        for player in self.game.players:
            block = QFrame()
            block.setFrameShape(QFrame.Shape.StyledPanel)
            block.setStyleSheet(f"background-color: {player.color}; border-radius: 8px;")
            v = QVBoxLayout()

            v.setContentsMargins(8, 0, 8, 0)  # left, top, right, bottom margins

            name_label = QLabel(f"{player.name}")
            name_label.setStyleSheet("color: white; font-weight: bold;")
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setFont(QFont("Arial", 12))
            points_label = QLabel(f"Points: {player.points}")
            road_label = QLabel(f"Longest road:{self._get_road_desc(player)}")
            army_label = QLabel(f"Largest army:{self._get_knight_desc(player)}")
            resources_label = QLabel(f"{self._get_resources_desc(player)}")

            v.addWidget(name_label)
            v.addWidget(points_label)
            v.addWidget(road_label)
            v.addWidget(army_label)
            v.addWidget(resources_label)

            block.setLayout(v)

            # Give each block a vertical stretch factor
            self.layout.addWidget(block, 1)  # the "1" here is the stretch factor

    def _get_road_desc(self, player):
        if self.game.has_player_the_longest_road(player):
            return "<span style='font-size:14px; font-weight:bold;'>✅</span>"
        else:
            return "<span style='font-size:14px; font-weight:bold;'>🚫</span>"
        
    def _get_knight_desc(self, player):
        if self.game.has_player_the_largest_army(player):
            return "<span style='font-size:14px; font-weight:bold;'>✅</span>"
        else:
            return "<span style='font-size:14px; font-weight:bold;'>🚫</span>"
        
    def _get_resources_desc(self, player):
        icons = ["🪵", "🧱", "🐑", "🌾", "🪨"]
        amounts = list(player.resources.values())

        # Make each entry 3 characters wide for alignment
        res_parts = [f"{icon}{str(amount)}"
                     for icon, amount in zip(icons, amounts)]
        res = " ".join(res_parts)
        return "<span style='font-size:16px; font-weight:bold;'>" + res + "</span>"

    def _clear_old_content(self):
        for i in reversed(range(self.layout.count())):
            w = self.layout.itemAt(i).widget()
            if w:
                w.deleteLater()
