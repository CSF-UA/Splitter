from typing import Optional
from PySide6.QtWidgets import QWidget, QDoubleSpinBox, QPushButton, QHBoxLayout

from .ui_component import UIComponent


class PeriodT0Section(UIComponent):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Period T0", parent)
        self.spin_t0 = None
        self.btn_t0_helper = None

    def create_widget(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.spin_t0 = QDoubleSpinBox()
        self.spin_t0.setRange(0.001, 1000.0)
        self.spin_t0.setDecimals(6)
        self.spin_t0.setSingleStep(0.1)
        layout.addWidget(self.spin_t0)

        self.btn_t0_helper = QPushButton("T0 [+]")
        self.btn_t0_helper.setCheckable(True)
        layout.addWidget(self.btn_t0_helper)

        return widget

    def update_state(self, t0_value: float = None):
        if t0_value is not None and self.spin_t0:
            self.spin_t0.setValue(t0_value)
