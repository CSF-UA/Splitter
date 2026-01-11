from typing import Optional
from PySide6.QtWidgets import QWidget, QRadioButton, QButtonGroup, QVBoxLayout

from src.models import Algorithm

from .ui_component import UIComponent


class AlgorithmSection(UIComponent):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Algorithm", parent)
        self.radio_algol = None
        self.radio_inverted_algol = None
        self.radio_normal = None
        self.type_group = None

    def create_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.radio_algol = QRadioButton(Algorithm.GB_AT.value)
        self.radio_algol.setChecked(True)
        layout.addWidget(self.radio_algol)

        self.radio_inverted_algol = QRadioButton(
            Algorithm.MAGNITUDE_INVERTED_GB_AT.value
        )
        layout.addWidget(self.radio_inverted_algol)

        self.radio_normal = QRadioButton(Algorithm.S_DIPS.value)
        layout.addWidget(self.radio_normal)

        self.type_group = QButtonGroup()
        self.type_group.addButton(self.radio_algol)
        self.type_group.addButton(self.radio_inverted_algol)
        self.type_group.addButton(self.radio_normal)

        return widget
