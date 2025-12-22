from PyQt6.QtWidgets import (
    QDialog, QGridLayout, QPushButton, QVBoxLayout, QLabel
)
from PyQt6.QtCore import Qt

from marl.ui.ChoiceOption import ChoiceOption


class ChoiceGridDialog(QDialog):
    def __init__(
        self,
        title: str,
        options: list[ChoiceOption],
        columns: int = 1,
        parent=None,
    ):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(360)

        main_layout = QVBoxLayout(self)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        main_layout.addWidget(title_label)

        grid = QGridLayout()
        main_layout.addLayout(grid)

        for i, option in enumerate(options):
            btn = QPushButton(option.text)
            btn.setEnabled(option.enabled)
            btn.setMinimumHeight(40)

            btn.setStyleSheet("""
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #888888;
                    border: 2px solid #999999;
                }
                QPushButton {
                    font-weight: bold;
                }
            """)

            row = i // columns
            col = i % columns

            def make_handler(opt=option):
                def handler():
                    opt.callback()
                    self.accept()
                return handler

            btn.clicked.connect(make_handler())
            grid.addWidget(btn, row, col)

        self.setLayout(main_layout)
