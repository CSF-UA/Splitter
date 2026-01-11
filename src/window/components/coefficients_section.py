from typing import Optional, Dict, Any, List, Union
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QHBoxLayout,
    QVBoxLayout,
)
from typing import Callable
from enum import Enum

from .ui_component import UIComponent
from ...models import Algorithm


class CoefficientType(Enum):
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"


class Coefficient:
    def __init__(
        self,
        key: str,
        label: str,
        coefficient_type: CoefficientType,
        default_value: Union[int, float, bool],
        min_value: Union[int, float] = 0,
        max_value: Union[int, float] = 100,
        decimals: int = 2,
        step: Union[int, float] = 1,
        description: str = "",
    ):
        self.key = key
        self.label = label
        self.coefficient_type = coefficient_type
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.decimals = decimals
        self.step = step
        self.description = description

    def create_widget(self) -> Union[QSpinBox, QDoubleSpinBox, QCheckBox]:
        """Create the appropriate widget for this coefficient."""
        if self.coefficient_type == CoefficientType.INTEGER:
            widget = QSpinBox()
            widget.setRange(int(self.min_value), int(self.max_value))
            widget.setValue(int(self.default_value))
            widget.setSingleStep(int(self.step))
        elif self.coefficient_type == CoefficientType.FLOAT:
            widget = QDoubleSpinBox()
            widget.setRange(float(self.min_value), float(self.max_value))
            widget.setValue(float(self.default_value))
            widget.setDecimals(self.decimals)
            widget.setSingleStep(float(self.step))
        else:  # BOOLEAN
            widget = QCheckBox()
            widget.setChecked(bool(self.default_value))
        return widget


# Coefficient definitions for each star type
# To add coefficients for a new star type:
# 1. Add the star type to Algorithm enum in models.py
# 2. Add an entry here with a list of Coefficient objects
# 3. Each Coefficient defines: key, label, type, default, min, max, and optional step/decimals
STAR_TYPE_COEFFICIENTS: Dict[Algorithm, List[Coefficient]] = {
    Algorithm.GB_AT: [
        Coefficient(
            key="algol_index_gap",
            label="Index Gap:",
            coefficient_type=CoefficientType.INTEGER,
            default_value=2,
            min_value=1,
            max_value=100,
            description="Gap between indices for Algol star analysis",
        ),
        Coefficient(
            key="cut_ratio",
            label="Cut Ratio:",
            coefficient_type=CoefficientType.FLOAT,
            default_value=0.25,
            min_value=0.01,
            max_value=0.99,
            step=0.05,
            description="Ratio for cutting data in Algol analysis",
        ),
        Coefficient(
            key="min_interval_points",
            label="Min Points:",
            coefficient_type=CoefficientType.INTEGER,
            default_value=5,
            min_value=1,
            max_value=100,
            description="Minimum points required for interval analysis",
        ),
        Coefficient(
            key="fill_remaining",
            label="Fill Remaining:",
            coefficient_type=CoefficientType.BOOLEAN,
            default_value=False,
            description="Fill all points not covered by main flow with intervals",
        ),
    ],
    Algorithm.MAGNITUDE_INVERTED_GB_AT: [
        Coefficient(
            key="algol_index_gap",
            label="Index Gap:",
            coefficient_type=CoefficientType.INTEGER,
            default_value=2,
            min_value=1,
            max_value=100,
            description="Gap between indices for inverted Algol star analysis",
        ),
        Coefficient(
            key="cut_ratio",
            label="Cut Ratio:",
            coefficient_type=CoefficientType.FLOAT,
            default_value=0.25,
            min_value=0.01,
            max_value=0.99,
            step=0.05,
            description="Ratio for cutting data in inverted Algol analysis",
        ),
        Coefficient(
            key="min_interval_points",
            label="Min Points:",
            coefficient_type=CoefficientType.INTEGER,
            default_value=5,
            min_value=1,
            max_value=100,
            description="Minimum points required for inverted interval analysis",
        ),
        Coefficient(
            key="fill_remaining",
            label="Fill Remaining:",
            coefficient_type=CoefficientType.BOOLEAN,
            default_value=False,
            description="Fill all points not covered by main flow with intervals",
        ),
    ],
    Algorithm.S_DIPS: [
        Coefficient(
            key="alpha",
            label="Alpha:",
            coefficient_type=CoefficientType.FLOAT,
            default_value=0.12,
            min_value=0.01,
            max_value=100.00,
            step=0.01,
            description="Alpha value for S-DIPS analysis",
        ),
    ],
}


def get_default_params_for_algorithm(
    algo_type: Algorithm,
) -> Dict[str, Union[int, float, bool]]:
    """Get default parameter values for a given algorithm type."""
    coefficients = STAR_TYPE_COEFFICIENTS.get(algo_type, [])
    return {coeff.key: coeff.default_value for coeff in coefficients}


class CoefficientsSection(UIComponent):
    """Coefficients configuration component."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        coefficient_callback: Optional[
            Callable[[str, Union[int, float, bool]], None]
        ] = None,
    ):
        super().__init__("Coefficients", parent)
        self.coefficient_callback = coefficient_callback
        self.current_star_type = Algorithm.GB_AT
        self.coefficient_widgets: Dict[
            str, Union[QSpinBox, QDoubleSpinBox, QCheckBox]
        ] = {}
        self.layouts: List[QHBoxLayout] = []
        self.main_layout = None

    def create_widget(self) -> QWidget:
        widget = QWidget()
        self.main_layout = QVBoxLayout(widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # Create coefficients for current star type
        self._create_coefficient_widgets()

        return widget

    def _create_coefficient_widgets(self):
        """Create coefficient widgets for the current star type."""
        # Clear existing widgets and layouts
        self._clear_coefficient_widgets()

        coefficients = STAR_TYPE_COEFFICIENTS[self.current_star_type]

        for coefficient in coefficients:
            layout = QHBoxLayout()
            layout.addWidget(QLabel(coefficient.label))

            widget = coefficient.create_widget()
            if coefficient.coefficient_type == CoefficientType.BOOLEAN:
                widget.stateChanged.connect(
                    lambda state, key=coefficient.key: self._on_coefficient_changed(
                        key, bool(state)
                    )
                )
            else:
                widget.valueChanged.connect(
                    lambda value, key=coefficient.key: self._on_coefficient_changed(
                        key, value
                    )
                )
            self.coefficient_widgets[coefficient.key] = widget
            layout.addWidget(widget)

            self.layouts.append(layout)
            self.main_layout.addLayout(layout)

    def _clear_coefficient_widgets(self):
        """Clear all coefficient widgets and layouts."""
        # Remove layouts from main layout
        for layout in self.layouts:
            # Remove widgets from layout
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
            self.main_layout.removeItem(layout)

        self.layouts.clear()
        self.coefficient_widgets.clear()

    def update_coefficients(self, params: Dict[str, Any]):
        """Update coefficient values."""
        for key, widget in self.coefficient_widgets.items():
            if key in params:
                widget.blockSignals(True)
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(params[key]))
                else:
                    widget.setValue(params[key])
                widget.blockSignals(False)

    def set_star_type(self, algo_type: Algorithm):
        """Change the star type and update the coefficient widgets."""
        if self.current_star_type == algo_type:
            return

        self.current_star_type = algo_type
        self._create_coefficient_widgets()

    def _on_coefficient_changed(self, key: str, value: Union[int, float, bool]):
        """Handle coefficient value changes."""
        if self.coefficient_callback:
            self.coefficient_callback(key, value)

    def get_coefficient_values(self) -> Dict[str, Union[int, float, bool]]:
        """Get current values of all coefficients."""
        result = {}
        for key, widget in self.coefficient_widgets.items():
            if isinstance(widget, QCheckBox):
                result[key] = widget.isChecked()
            else:
                result[key] = widget.value()
        return result
