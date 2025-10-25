import sys
import os
from PyQt6.QtWidgets import QApplication

from marl.ui.MainWindow import MainWindow

# Fix for Wayland/X11 if needed
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
