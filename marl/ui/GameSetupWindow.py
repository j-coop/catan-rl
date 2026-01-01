from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QCheckBox
)

class GameSetupWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Game Setup")

        self.player_rows = []
        layout = QVBoxLayout(self)

        names = ["Blue Player", "Purple Player", "Yellow Player", "Green Player"]

        for name in names:
            row = QHBoxLayout()
            label = QLabel(name)
            checkbox = QCheckBox("AI Controlled")
            row.addWidget(label)
            row.addWidget(checkbox)
            layout.addLayout(row)
            self.player_rows.append((name, checkbox))

        self.start_btn = QPushButton("Start Game")
        self.start_btn.clicked.connect(self.start_game)
        layout.addWidget(self.start_btn)

    def start_game(self):
        config = {
            name: checkbox.isChecked()
            for name, checkbox in self.player_rows
        }
        self.launch_game(config)
