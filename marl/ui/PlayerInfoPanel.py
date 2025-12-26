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
        self.player_rows = {}
        self.refresh()

    def _is_active_player(self, player):
        return player == self.game.current_player

    def _get_road_desc(self, player):
        if self.game.has_player_the_longest_road(player):
            return f"<span style='font-size:14px; font-weight:bold;'>✅ {self.game.longest_road_length}</span>"
        else:
            return (f"<span style='font-size:14px; font-weight:bold;'>"
                    f"🚫 {player.longest_road}/{max(5, self.game.longest_road_length)}"
                    f"</span>")

    def _get_knight_desc(self, player):
        if self.game.has_player_the_largest_army(player):
            return f"<span style='font-size:14px; font-weight:bold;'>✅ {self.game.largest_army_count}</span>"
        else:
            return (f"<span style='font-size:14px; font-weight:bold;'>"
                    f"🚫 {player.knights_played}/{self.game.largest_army_count}"
                    f"</span>")

    def _get_resources_desc(self, player):
        icons = ["🪵", "🧱", "🐑", "🌾", "🪨"]
        amounts = list(player.resources.values())
        res_parts = [f"{icon}{amount}" for icon, amount in zip(icons, amounts)]
        return "<span style='font-size:16px; font-weight:bold;'>" + " ".join(res_parts) + "</span>"

    def _clear_old_content(self):
        for i in reversed(range(self.layout.count())):
            w = self.layout.itemAt(i).widget()
            if w:
                w.deleteLater()

    def refresh(self):
        self._clear_old_content()
        self.player_rows.clear()

        for player in self.game.players:
            is_active = self._is_active_player(player)

            block = QFrame()
            block.setObjectName("playerBlock")
            block.setFrameShape(QFrame.Shape.NoFrame)

            if is_active:
                block.setStyleSheet(
                    f"""
                    QFrame#playerBlock {{
                        background-color: {player.color};
                        border-radius: 8px;
                        border: 3px solid black;
                    }}
                    """
                )
            else:
                block.setStyleSheet(
                    f"""
                    QFrame#playerBlock {{
                        background-color: {player.color};
                        border-radius: 8px;
                        border: none;
                    }}
                    """
                )

            v = QVBoxLayout(block)
            v.setContentsMargins(8, 4, 8, 4)

            dot = "● " if is_active else ""
            name_label = QLabel(f"{dot}{player.name}")
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            if is_active:
                name_label.setFont(QFont("Arial", 14, QFont.Weight.Black))
                name_label.setStyleSheet("color: white; font-style: italic;")
            else:
                name_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
                name_label.setStyleSheet("color: white;")

            points_label = QLabel()
            road_label = QLabel()
            army_label = QLabel()
            resources_label = QLabel()

            v.addWidget(name_label)
            v.addWidget(points_label)
            v.addWidget(road_label)
            v.addWidget(army_label)
            v.addWidget(resources_label)

            self.layout.addWidget(block, 1)

            self.player_rows[player.name] = {
                "points": points_label,
                "road": road_label,
                "army": army_label,
                "resources": resources_label,
            }

        self.update_all()

    def update_all(self):
        for player in self.game.players:
            row = self.player_rows[player.name]

            row["points"].setText(f"Points: {player.points}")
            row["road"].setText(
                f"Longest road: {self._get_road_desc(player)}"
            )
            row["army"].setText(
                f"Largest army: {self._get_knight_desc(player)}"
            )
            row["resources"].setText(
                self._get_resources_desc(player)
            )

    def _update_after_game_change(self):
        """Call after ANY game action that may affect UI state."""
        self.update_all()
