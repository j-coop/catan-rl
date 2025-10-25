# ui/main_window.py
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton

class MainWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Settlers of Catan (RL Testing)")
        layout = QVBoxLayout()
        self.build_road_btn = QPushButton("Build Road")
        self.build_road_btn.clicked.connect(self.on_build_road)
        layout.addWidget(self.build_road_btn)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def render_state(self, state):
        # Refresh board + player stats
        pass

    def on_build_road(self):
        # Example dummy action
        action = {"type": "build_road", "player": 1, "location": (0, 1)}
        self.controller.handle_action(action)
