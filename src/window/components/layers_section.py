from typing import Optional
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout
from PySide6.QtCore import Qt

from .ui_component import UIComponent


class LayersSection(UIComponent):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Layers", parent)
        self.lbl_layer_status = None
        self.btn_layer_prev = None
        self.btn_layer_next = None
        self.btn_layer_add = None
        self.btn_layer_rem = None

    def create_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_layer_status = QLabel("Layer 1 / 1")
        self.lbl_layer_status.setAlignment(Qt.AlignCenter)
        self.lbl_layer_status.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(self.lbl_layer_status)

        button_layout = QHBoxLayout()

        self.btn_layer_prev = QPushButton("◀")
        self.btn_layer_prev.setFixedWidth(40)
        button_layout.addWidget(self.btn_layer_prev)

        self.btn_layer_next = QPushButton("▶")
        self.btn_layer_next.setFixedWidth(40)
        button_layout.addWidget(self.btn_layer_next)

        button_layout.addStretch()

        self.btn_layer_add = QPushButton("+")
        self.btn_layer_add.setFixedWidth(40)
        self.btn_layer_add.setStyleSheet("font-weight: bold;")
        button_layout.addWidget(self.btn_layer_add)

        self.btn_layer_rem = QPushButton("−")
        self.btn_layer_rem.setFixedWidth(40)
        self.btn_layer_rem.setStyleSheet("font-weight: bold;")
        button_layout.addWidget(self.btn_layer_rem)

        layout.addLayout(button_layout)

        return widget

    def update_layer_status(self, current: int, total: int):
        if self.lbl_layer_status:
            self.lbl_layer_status.setText(f"Layer {current + 1} / {total}")
