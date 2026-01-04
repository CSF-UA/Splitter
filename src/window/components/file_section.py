from typing import Optional
from PySide6.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout

from .ui_component import UIComponent
from src.contants import COLORS


class FileSection(UIComponent):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("File", parent)
        self.btn_browse = None
        self.lbl_filename = None

    def create_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn_browse = QPushButton("Open File...")
        layout.addWidget(self.btn_browse)

        self.lbl_filename = QLabel("No file selected")
        self.lbl_filename.setWordWrap(True)
        self.lbl_filename.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 11px;"
        )
        layout.addWidget(self.lbl_filename)

        return widget
