# ui/game_controller.py
from PyQt6.QtWidgets import QApplication

from marl.model.CatanGame import CatanGame
from marl.ui.MainWindow import MainWindow


class GameController:
    def __init__(self):
        self.app = QApplication([])
        self.window = MainWindow(self)
        self.state = CatanGame(player_colors=['blue', 'red', 'black', 'white'])
        self.window.render_state(self.state)

    def start(self):
        self.window.show()
        self.app.exec()

    def handle_action(self, action):
        # Apply logic
        self.state.apply(action)
        self.window.render_state(self.state)
