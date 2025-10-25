# main.py
from marl.ui.GameController import GameController

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

if __name__ == "__main__":
    controller = GameController()
    controller.start()
