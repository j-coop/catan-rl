import sys
import os
from PyQt6.QtWidgets import QApplication

from marl.model.CatanGame import CatanGame
from marl.ui.CatanWindow import CatanWindow

# Fix for Wayland/X11 if needed
os.environ.setdefault("QT_QPA_PLATFORM", "windows")

def main():
    app = QApplication(sys.argv)

    colors=["#2c3aff", "#c525c5", "#dbc33a", '#32a852']
    names = ["Blue Player", "Purple Player", "Yellow Player", "Green Player"]
    game = CatanGame(player_colors=colors, player_names=names)
    window = CatanWindow(game)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
