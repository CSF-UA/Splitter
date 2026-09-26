import sys
from pathlib import Path


def main():
    if "--batch" in sys.argv:  # headless: uv run main.py --batch [--period P] FILES...
        from splitter.auto import batch

        args = [a for a in sys.argv[1:] if a != "--batch"]
        period = 0.0
        if "--period" in args:
            i = args.index("--period")
            period = float(args[i + 1])
            del args[i : i + 2]
        return batch(args, period)

    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication
    from vispy.app import use_app

    use_app("pyside6")
    from splitter.window import SplitterWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font_id = QFontDatabase.addApplicationFont(str(Path(__file__).parent / "fonts" / "inter.ttf"))
    families = QFontDatabase.applicationFontFamilies(font_id) if font_id != -1 else []
    app.setFont(QFont(families[0] if families else "Inter", 10))
    window = SplitterWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
