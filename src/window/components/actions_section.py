from typing import Optional
from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout

from .ui_component import UIComponent
from src.contants import COLORS


class ActionsSection(UIComponent):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Actions", parent)
        self.btn_compute = None
        self.btn_remove = None
        self.btn_save = None
        self.btn_reset = None

    def create_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn_compute = QPushButton("Load + Compute")
        self.btn_compute.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                font-weight: bold;
                padding: 8px;
            }}
            QPushButton:hover {{
                background-color: #22c55e;
            }}
        """)
        layout.addWidget(self.btn_compute)

        self.btn_remove = QPushButton("Remove Selected")
        layout.addWidget(self.btn_remove)

        self.btn_save = QPushButton("Save (.da!)")
        layout.addWidget(self.btn_save)

        self.btn_reset = QPushButton("Reset View")
        layout.addWidget(self.btn_reset)

        return widget
