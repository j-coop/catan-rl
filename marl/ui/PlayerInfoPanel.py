from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame

from marl.model.CatanGame import CatanGame


class PlayerInfoPanel(QWidget):
    """Left-side panel showing player stats."""

    def __init__(self, game: CatanGame, config):
        super().__init__()
        self.game = game
        self.config = config  # player_name: choice (Human/AI/Bot) format
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.setFixedWidth(230)
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
                    f"🚫 {player.knights_played}/{max(3, self.game.largest_army_count)}"
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

            game_over = self.game.game_over

            if is_active:
                block.setStyleSheet(
                    f"""
                    QFrame#playerBlock {{
                        background-color: {player.color if not game_over or self.game.winner == player.name else 'grey'};
                        border-radius: 8px;
                        border: 3px solid black;
                    }}
                    """
                )
            else:
                block.setStyleSheet(
                    f"""
                    QFrame#playerBlock {{
                        background-color: {player.color if not game_over or self.game.winner == player.name else 'grey'};
                        border-radius: 8px;
                        border: none;
                    }}
                    """
                )

            v = QVBoxLayout(block)
            v.setContentsMargins(8, 4, 8, 4)

            dot = "● " if is_active else ""
            
            choice = self.config.get(player.name, "Human")
            if choice == "AI":
                icon = " 🧠"
                label = " AI"
            elif "Bot" in choice:
                icon = " 🤖"
                level = choice.split()[-1]
                label = f" BOT ({level})"
            else:
                icon = " 👤"
                label = "HUMAN"

            clean_name = player.name.replace(' Player', '')
            if is_active:
                display_text = f"{dot}<i>{clean_name}</i>{icon}{label}"
                name_label = QLabel(display_text)
                name_label.setFont(QFont("Arial", 14, QFont.Weight.Black))
            else:
                display_text = f"{dot}{clean_name}{icon}{label}"
                name_label = QLabel(display_text)
                name_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))

            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
