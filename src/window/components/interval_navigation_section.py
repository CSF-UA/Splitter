from typing import Optional
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout
from PySide6.QtCore import Qt

from .ui_component import UIComponent


class IntervalNavigationSection(UIComponent):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Interval Navigation", parent)
        self.lbl_interval_status = None
        self.btn_interval_prev = None
        self.btn_interval_next = None

    def create_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_interval_status = QLabel("No intervals")
        self.lbl_interval_status.setAlignment(Qt.AlignCenter)
        self.lbl_interval_status.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(self.lbl_interval_status)

        button_layout = QHBoxLayout()

        self.btn_interval_prev = QPushButton("◀")
        self.btn_interval_prev.setFixedWidth(40)
        button_layout.addWidget(self.btn_interval_prev)

        self.btn_interval_next = QPushButton("▶")
        self.btn_interval_next.setFixedWidth(40)
        button_layout.addWidget(self.btn_interval_next)

        layout.addLayout(button_layout)

        return widget

    def update_interval_status(self, total_intervals: int, current_idx: int = -1):
        if (
            not self.lbl_interval_status
            or not self.btn_interval_prev
            or not self.btn_interval_next
        ):
            return

        if total_intervals == 0:
            self.lbl_interval_status.setText("No intervals")
            self.btn_interval_prev.setEnabled(False)
            self.btn_interval_next.setEnabled(False)
            return

        if current_idx >= 0:
            self.lbl_interval_status.setText(
                f"Interval {current_idx + 1} / {total_intervals}"
            )
        else:
            self.lbl_interval_status.setText(f"{total_intervals} intervals")

        # Enable both buttons for circular navigation
        self.btn_interval_prev.setEnabled(True)
        self.btn_interval_next.setEnabled(True)
