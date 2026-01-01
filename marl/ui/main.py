import platform
import sys
import os
from PyQt6.QtWidgets import QApplication

from marl.model.CatanGame import CatanGame
from marl.ui.CatanWindow import CatanWindow
from marl.ui.GameSetupWindow import GameSetupWindow

# Fix for Wayland/X11 if needed
if platform.system() == "Windows":
    os.environ.setdefault("QT_QPA_PLATFORM", "windows")
elif platform.system() == "Linux":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

def main():
    app = QApplication(sys.argv)

    setup = GameSetupWindow()
    setup.show()

    def launch(config):
        colors = ["#2c3aff", "#c525c5", "#dbc33a", '#32a852']
        names = list(config.keys())

        game = CatanGame(
            player_colors=colors,
            player_names=names,
            init_placement_model_path="models/init_placement_model.zip",
        )

        app.main_window = CatanWindow(game)  # Keep reference for window to open
        app.main_window.show()

        setup.close()

    setup.launch_game = launch
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
