import platform
import sys
import os
from PyQt6.QtWidgets import QApplication

from marl.env.ActionSpace import ActionSpace
from marl.env.tianshou.multi_agent_env import CatanEnv
from marl.model.CatanGame import CatanGame
from marl.model.GameManager import GameManager
from marl.ui.CatanWindow import CatanWindow
from marl.ui.EnvMock import EnvMock
from marl.ui.GameSetupWindow import GameSetupWindow
from marl.ui.controllers.HumanController import HumanController
from marl.ui.controllers.AgentController import AgentController

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

        controllers = {}
        for name, is_ai in config.items():
            if is_ai:
                controllers[name] = AgentController(name, name)
            else:
                controllers[name] = HumanController(name)

        env = CatanEnv()
        env.game = game
        action_space = ActionSpace(env)
        action_space.init_action_specs(use_callbacks=True)

        game_manager = GameManager(game, controllers, action_space, config)

        app.main_window = CatanWindow(game, game_manager)  # Keep reference for window to open
        app.main_window.show()

        game_manager.on_turn_changed()

        setup.close()

    setup.launch_game = launch
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
