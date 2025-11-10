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

        player_count = len(self.game.players)
        if player_count == 0:
            return

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
            v.addWidget(name_label)

            points_label = QLabel(f"Points: {player.points}")
            v.addWidget(points_label)

            road_label = QLabel(f"Longest road:{self._get_road_desc(player)}")
            v.addWidget(road_label)

            army_label = QLabel(f"Largest army:{self._get_knight_desc(player)}")
            v.addWidget(army_label)

            res_summary = ", ".join(f"{r[0].upper()}:{r[1]}" for r in player.resources.items())
            resources_label = QLabel(f"Resources: {res_summary}")
            v.addWidget(resources_label)

            block.setLayout(v)

            # Give each block a vertical stretch factor
            self.layout.addWidget(block, 1)  # the "1" here is the stretch factor

    def _get_road_desc(self, player):
        if self.game.has_player_the_longest_road(player):
            return "<span style='font-size:16px; font-weight:bold;'>✅</span>"
        else:
            return "<span style='font-size:16px; font-weight:bold;'>🚫</span>"
        
    def _get_knight_desc(self, player):
        if self.game.has_player_the_largest_army(player):
            return "<span style='font-size:16px; font-weight:bold;'>✅</span>"
        else:
            return "<span style='font-size:16px; font-weight:bold;'>🚫</span>"
