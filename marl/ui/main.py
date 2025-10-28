import sys
import os
from PyQt6.QtWidgets import QApplication

from marl.model.CatanGame import CatanGame
from marl.ui.CatanWindow import CatanWindow

# Fix for Wayland/X11 if needed
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

def main():
    app = QApplication(sys.argv)
    game = CatanGame(player_colors=['blue', 'red', 'green', 'purple'])
    window = CatanWindow(game)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
