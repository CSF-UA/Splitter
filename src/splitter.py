import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication,
)
from PySide6.QtGui import QFont, QFontDatabase

from vispy.app import use_app

from src.window.layout import SplitterWindow

use_app("pyside6")


def _load_font():
    current_dir = Path(__file__).parent.parent
    font_path = current_dir / "fonts" / "inter.ttf"

    font_family = "Inter"

    if font_path.exists():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id != -1:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                font_family = families[0]
    else:
        available_fonts = QFontDatabase().families()
        if "Inter" not in available_fonts:
            fallback_options = [
                "SF Pro Display",
                "Roboto",
                "Arial",
                "Helvetica",
                "Segoe UI",
            ]
            for fallback in fallback_options:
                if fallback in available_fonts:
                    font_family = fallback
                    break

    return QFont(font_family, 10)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    font = _load_font()
    app.setFont(font)

    window = SplitterWindow()
    window.show()

    sys.exit(app.exec())
