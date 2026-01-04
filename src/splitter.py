import sys
from PySide6.QtWidgets import (
    QApplication,
)
from PySide6.QtGui import QFont

from vispy.app import use_app

from src.window.layout import SplitterWindow

use_app("pyside6")

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = SplitterWindow()
    window.show()

    sys.exit(app.exec())
