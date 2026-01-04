from typing import Optional, Callable
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QDoubleSpinBox,
)
from PySide6.QtCore import Qt

from .ui_component import UIComponent


class ZoomSection(UIComponent):
    """Zoom controls for lightcurve visualization."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        time_zoom_callback: Optional[Callable[[float], None]] = None,
        magnitude_zoom_callback: Optional[Callable[[float], None]] = None,
        reset_callback: Optional[Callable[[], None]] = None,
    ):
        super().__init__("Zoom", parent)
        self.time_zoom_callback = time_zoom_callback
        self.magnitude_zoom_callback = magnitude_zoom_callback
        self.reset_callback = reset_callback

        # Zoom parameters
        self.x_zoom = 1.0  # X: time axis zoom factor
        self.y_zoom = 1.0  # Y: magnitude axis zoom factor

    def create_widget(self) -> QWidget:
        """Create the zoom control widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # X parameter section
        x_group = self._create_parameter_section(
            "X", 0.1, 100.0, 1.0, 
            self._on_x_slider_changed,
            self._update_x_value
        )
        self.x_slider = x_group['slider']
        self.x_display = x_group['display']
        self.x_plus_btn = x_group['plus_btn']
        self.x_minus_btn = x_group['minus_btn']
        self.x_reset_btn = x_group['reset_btn']
        layout.addLayout(x_group['layout'])

        # Y parameter section
        y_group = self._create_parameter_section(
            "Y", 0.1, 100.0, 1.0,
            self._on_y_slider_changed,
            self._update_y_value
        )
        self.y_slider = y_group['slider']
        self.y_display = y_group['display']
        self.y_plus_btn = y_group['plus_btn']
        self.y_minus_btn = y_group['minus_btn']
        self.y_reset_btn = y_group['reset_btn']
        layout.addLayout(y_group['layout'])

        return widget

    def _create_parameter_section(self, label_text: str, min_val: float, max_val: float, default_val: float, slider_callback, value_callback):
        """Create a parameter control section with label, display, slider, and buttons."""
        # Label and display
        label = QLabel(f"{label_text}:")
        display = QDoubleSpinBox()
        display.setRange(min_val, max_val)
        display.setValue(default_val)
        display.setDecimals(2)
        display.setSingleStep(0.1)
        display.setReadOnly(True)  # Display only, controlled by slider

        header_layout = QHBoxLayout()
        header_layout.addWidget(label)
        header_layout.addWidget(display)

        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setRange(1, 1000)
        slider.setValue(self._value_to_slider(default_val, min_val, max_val))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(100)
        slider.valueChanged.connect(slider_callback)

        # Control buttons
        plus_btn = QPushButton("+")
        plus_btn.setFixedWidth(40)
        plus_btn.clicked.connect(lambda: self._adjust_parameter(display, slider, 0.1, min_val, max_val, value_callback))

        minus_btn = QPushButton("-")
        minus_btn.setFixedWidth(40)
        minus_btn.clicked.connect(lambda: self._adjust_parameter(display, slider, -0.1, min_val, max_val, value_callback))

        reset_btn = QPushButton("⟳")
        reset_btn.setFixedWidth(40)
        reset_btn.clicked.connect(lambda: self._reset_parameter(display, slider, default_val, min_val, max_val, value_callback))

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        buttons_layout.addWidget(plus_btn)
        buttons_layout.addWidget(minus_btn)
        buttons_layout.addWidget(reset_btn)
        buttons_layout.addStretch()

        # Main layout for this parameter
        param_layout = QVBoxLayout()
        param_layout.addLayout(header_layout)
        param_layout.addWidget(slider)
        param_layout.addLayout(buttons_layout)

        return {
            'layout': param_layout,
            'display': display,
            'slider': slider,
            'plus_btn': plus_btn,
            'minus_btn': minus_btn,
            'reset_btn': reset_btn
        }

    def _value_to_slider(self, value: float, min_val: float, max_val: float) -> int:
        """Convert parameter value to slider position."""
        # Map value range to 1-1000 slider range
        range_size = max_val - min_val
        if range_size == 0:
            return 500  # Middle value
        normalized = (value - min_val) / range_size
        return int(1 + normalized * 999)

    def _slider_to_value(self, slider_value: int, min_val: float, max_val: float) -> float:
        """Convert slider position to parameter value."""
        normalized = (slider_value - 1) / 999.0
        return min_val + normalized * (max_val - min_val)

    def _adjust_parameter(self, display, slider, delta: float, min_val: float, max_val: float, value_callback):
        """Adjust parameter value by delta."""
        current_value = display.value()
        new_value = max(min_val, min(max_val, current_value + delta))
        slider_pos = self._value_to_slider(new_value, min_val, max_val)
        
        # Update display and slider
        display.setValue(new_value)
        slider.blockSignals(True)
        slider.setValue(slider_pos)
        slider.blockSignals(False)
        
        # Update internal state and external callback directly with exact value
        # to avoid precision loss from round-trip conversion
        value_callback(new_value)

    def _reset_parameter(self, display, slider, default_val: float, min_val: float, max_val: float, value_callback):
        """Reset parameter to default value."""
        display.setValue(default_val)
        slider.blockSignals(True)
        slider.setValue(self._value_to_slider(default_val, min_val, max_val))
        slider.blockSignals(False)
        value_callback(default_val)

    def _update_x_value(self, value: float):
        """Update X zoom value and trigger callback."""
        self.x_zoom = value
        if self.time_zoom_callback:
            self.time_zoom_callback(value)

    def _update_y_value(self, value: float):
        """Update Y zoom value and trigger callback."""
        self.y_zoom = value
        if self.magnitude_zoom_callback:
            self.magnitude_zoom_callback(value)

    def _on_x_slider_changed(self, slider_value: int):
        """Handle X slider value change (time axis zoom)."""
        zoom_factor = self._slider_to_value(slider_value, 0.1, 100.0)
        self.x_display.setValue(zoom_factor)
        self._update_x_value(zoom_factor)

    def _on_y_slider_changed(self, slider_value: int):
        """Handle Y slider value change (magnitude axis zoom)."""
        zoom_factor = self._slider_to_value(slider_value, 0.1, 100.0)
        self.y_display.setValue(zoom_factor)
        self._update_y_value(zoom_factor)


    def set_x_zoom(self, x_zoom: float):
        """Set the X zoom factor programmatically (time axis)."""
        self.x_zoom = x_zoom
        self.x_display.setValue(x_zoom)
        self.x_slider.blockSignals(True)
        self.x_slider.setValue(self._value_to_slider(x_zoom, 0.1, 100.0))
        self.x_slider.blockSignals(False)

        if self.time_zoom_callback:
            self.time_zoom_callback(x_zoom)

    def set_y_zoom(self, y_zoom: float):
        """Set the Y zoom factor programmatically (magnitude axis)."""
        self.y_zoom = y_zoom
        self.y_display.setValue(y_zoom)
        self.y_slider.blockSignals(True)
        self.y_slider.setValue(self._value_to_slider(y_zoom, 0.1, 100.0))
        self.y_slider.blockSignals(False)

        if self.magnitude_zoom_callback:
            self.magnitude_zoom_callback(y_zoom)

    def get_x_zoom(self) -> float:
        """Get the X zoom factor."""
        return self.x_zoom

    def get_y_zoom(self) -> float:
        """Get the Y zoom parameter."""
        return self.y_zoom
