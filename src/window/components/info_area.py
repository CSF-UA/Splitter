from typing import Optional
from PySide6.QtWidgets import QLabel, QWidget

from .ui_component import UIComponent
from src.contants import COLORS


class InfoArea(UIComponent):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("", parent)  # No title for info area
        self.info_label = None

    def create_widget(self) -> QLabel:
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(f"""
            background-color: {COLORS["bg_panel"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 6px;
            padding: 10px;
            color: {COLORS["text"]};
            font-size: 11px;
        """)
        self.info_label.setMinimumHeight(100)

        return self.info_label

    def set_info(self, message: str):
        """Set the info message."""
        if self.info_label:
            self.info_label.setText(message)
