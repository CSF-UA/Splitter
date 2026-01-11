import os
import traceback
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStatusBar,
    QSizePolicy,
    QFileDialog,
)
from PySide6.QtGui import QAction, QKeySequence
import numpy as np

from src.models import Algorithm
from src.contants import COLORS
from src.canvas import LightcurveCanvas
from src.core import (
    get_data,
    save_data,
    splitting_algol_configurable,
    splitting_normal,
    check_up,
)
from src.window.components import (
    FileSection,
    AlgorithmSection,
    PeriodT0Section,
    ActionsSection,
    InfoArea,
    LayersSection,
    CoefficientsSection,
    IntervalNavigationSection,
    ZoomSection,
)
from src.window.components.coefficients_section import get_default_params_for_algorithm


class SplitterWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.fname: str = ""
        self.T0: float = 0.0
        self.algo_type: Algorithm = Algorithm.GB_AT

        self.JD: np.ndarray = np.array([])
        self.mag: np.ndarray = np.array([])

        # Layers structure
        self._reset_layers()

        # UI Components
        self.file_section = FileSection(self)
        self.algorithm_section = AlgorithmSection(self)
        self.t0_section = PeriodT0Section(self)
        self.actions_section = ActionsSection(self)
        self.info_area = InfoArea(self)
        self.layers_section = LayersSection(self)
        self.coefficients_section = CoefficientsSection(
            self, self._on_coefficient_change
        )
        self.interval_nav_section = IntervalNavigationSection(self)
        self.zoom_section = ZoomSection(
            self,
            self._on_time_zoom_change,
            self._on_magnitude_zoom_change,
            self._on_zoom_reset,
        )

        self._setup_ui()
        self._setup_shortcuts()
        self._apply_style()

    def _reset_layers(self):
        """Reset layers to initial state (single default layer)."""
        self.layers = [
            {
                "params": get_default_params_for_algorithm(self.algo_type),
                "intervals": {"start": [], "finish": []},
            }
        ]
        self.current_layer_idx = 0

    def _setup_ui(self):
        """Build the user interface."""
        self.setWindowTitle("Splitter v4.0.1")
        self.setMinimumSize(1400, 800)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # === LEFT PANEL ===
        left_panel = QWidget()
        left_panel.setFixedWidth(220)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Add components to left panel
        left_layout.addWidget(self.file_section.get_widget())
        left_layout.addWidget(self.algorithm_section.get_widget())
        left_layout.addWidget(self.t0_section.get_widget())
        left_layout.addWidget(self.actions_section.get_widget())
        left_layout.addWidget(self.info_area.get_widget())

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # === CENTER (VisPy Canvas) ===
        self.canvas = LightcurveCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.interval_selected.connect(self._on_interval_selected)
        self.canvas.point_clicked.connect(self._on_point_clicked)
        self.canvas.t0_updated.connect(self._update_t0_from_helper)
        main_layout.addWidget(self.canvas, stretch=1)

        # === RIGHT PANEL ===
        right_panel = QWidget()
        right_panel.setFixedWidth(200)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        # Add components to right panel
        right_layout.addWidget(self.layers_section.get_widget())
        right_layout.addWidget(self.interval_nav_section.get_widget())
        right_layout.addWidget(self.coefficients_section.get_widget())
        right_layout.addWidget(self.zoom_section.get_widget())
        right_layout.addStretch()

        main_layout.addWidget(right_panel)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Connect component signals
        self._connect_component_signals()

        # Initialize component states
        self._initialize_components()

    def _connect_component_signals(self):
        """Connect all component signals to window slots."""
        # File section
        self.file_section.btn_browse.clicked.connect(self._browse_file)

        # Star type section
        self.algorithm_section.type_group.buttonClicked.connect(self._on_type_change)

        # T0 section
        self.t0_section.spin_t0.setValue(self.T0)
        self.t0_section.spin_t0.valueChanged.connect(self._on_t0_change)
        self.t0_section.btn_t0_helper.clicked.connect(self._toggle_t0_helper)

        # Actions section
        self.actions_section.btn_compute.clicked.connect(self._load_and_compute)
        self.actions_section.btn_remove.clicked.connect(self._remove_selected)
        self.actions_section.btn_save.clicked.connect(self._save_intervals)
        self.actions_section.btn_reset.clicked.connect(self._reset_view)

        # Layers section
        self.layers_section.btn_layer_prev.clicked.connect(self._layer_prev)
        self.layers_section.btn_layer_next.clicked.connect(self._layer_next)
        self.layers_section.btn_layer_add.clicked.connect(self._layer_add)
        self.layers_section.btn_layer_rem.clicked.connect(self._layer_remove)

        # Interval navigation section
        self.interval_nav_section.btn_interval_prev.clicked.connect(self._interval_prev)
        self.interval_nav_section.btn_interval_next.clicked.connect(self._interval_next)

    def _initialize_components(self):
        """Initialize component states."""
        # Set initial T0 value
        self.t0_section.update_state(t0_value=self.T0)

        # Show period section only if S-DIPS is selected (hidden by default since default is GB_AT)
        if self.algo_type == Algorithm.S_DIPS:
            self.t0_section.get_widget().setVisible(True)
        else:
            self.t0_section.get_widget().setVisible(False)

        # Set initial coefficients
        self.coefficients_section.update_coefficients(self.layers[0]["params"])

        # Set initial layer status
        self.layers_section.update_layer_status(0, len(self.layers))

        # Disable time expansion section initially (no data loaded)
        self.zoom_section.set_enabled(False)

        # Set initial info message
        self.info_area.set_info(
            "Welcome!\n\n1. Open a .tess file\n2. Adjust T0 if needed\n3. Click 'Load + Compute'\n\nShortcuts: D=delete, S=save, R=reset"
        )

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Delete shortcut
        delete_action = QAction(self)
        delete_action.setShortcut(QKeySequence("D"))
        delete_action.triggered.connect(self._remove_selected)
        self.addAction(delete_action)

        # Save shortcut
        save_action = QAction(self)
        save_action.setShortcut(QKeySequence("S"))
        save_action.triggered.connect(self._save_intervals)
        self.addAction(save_action)

        # Reset view shortcut
        reset_action = QAction(self)
        reset_action.setShortcut(QKeySequence("R"))
        reset_action.triggered.connect(self._reset_view)
        self.addAction(reset_action)

        # Escape to cancel T0 helper
        escape_action = QAction(self)
        escape_action.setShortcut(QKeySequence("Escape"))
        escape_action.triggered.connect(self._cancel_all_modes)
        self.addAction(escape_action)

    def _apply_style(self):
        """Apply light theme stylesheet matching v4.0."""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS["bg_dark"]};
            }}
            QWidget {{
                color: {COLORS["text"]};
                font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
                font-size: 12px;
            }}
            QGroupBox {{
                background-color: {COLORS["bg_panel"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 8px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {COLORS["accent"]};
            }}
            QPushButton {{
                background-color: {COLORS["bg_button"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 6px;
                padding: 6px 12px;
                color: {COLORS["text"]};
            }}
            QPushButton:hover {{
                background-color: {COLORS["bg_button_hover"]};
            }}
            QPushButton:pressed {{
                background-color: {COLORS["accent"]};
                color: white;
            }}
            QPushButton:checked {{
                background-color: {COLORS["accent"]};
                color: white;
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox {{
                background-color: {COLORS["bg_panel"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 4px;
                padding: 4px 8px;
                color: {COLORS["text"]};
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border-color: {COLORS["accent"]};
            }}
            QRadioButton {{
                spacing: 8px;
                color: {COLORS["text"]};
            }}
            QRadioButton::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 8px;
                border: 2px solid {COLORS["border"]};
                background-color: {COLORS["bg_panel"]};
            }}
            QRadioButton::indicator:checked {{
                background-color: {COLORS["accent"]};
                border-color: {COLORS["accent"]};
            }}
            QStatusBar {{
                background-color: {COLORS["bg_panel"]};
                color: {COLORS["text_dim"]};
                border-top: 1px solid {COLORS["border"]};
            }}
            QLabel {{
                background-color: transparent;
            }}
        """)

    def _set_info(self, msg: str):
        """Update the info label."""
        self.info_area.set_info(msg)

    def _browse_file(self):
        """Open file dialog to select a .tess file."""
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select .tess file", "", "TESS/LC files (*.tess);;All files (*.*)"
        )
        if fname:
            self.fname = fname
            self.file_section.lbl_filename.setText(os.path.basename(fname))
            self._load_preview()

    def _load_preview(self):
        """Load and display lightcurve preview."""
        if not self.fname or not os.path.exists(self.fname):
            return

        try:
            JD, mag = get_data(self.fname, -1000, 1000)
            if len(JD) == 0:
                self._set_info("File loaded, but no valid data points.")
                return

            self.JD, self.mag = JD, mag

            # Reset intervals for all layers
            for layer in self.layers:
                layer["intervals"] = {"start": [], "finish": []}

            # Reset period after file is loaded
            self.T0 = 0.0
            self.t0_section.update_state(t0_value=0.0)
            self.t0_section.set_error_state(False)

            # Update canvas with title
            self.canvas.set_title(self.fname, self.algo_type)
            self.canvas.set_data(self.JD, self.mag)
            self.canvas.set_layers(self.layers)
            self._update_interval_navigation_status()
            self._reset_layers()
            self._update_layer_ui()
            # Enable time expansion section now that we have data
            self.zoom_section.set_enabled(True)

            self._set_info(
                f"Loaded {len(self.JD)} points\n\n"
                f"Adjust T0 and coefficients,\n"
                f"then click 'Load + Compute'\n"
                f"to detect intervals."
            )
            self.status_bar.showMessage(f"Loaded: {os.path.basename(self.fname)}")

        except Exception as e:
            self._set_info(f"Preview failed: {e}")

    def _on_type_change(self, button):
        """Handle star type change."""
        self.algo_type = Algorithm(
            button.text()
        )  # Extract "Algol" or "Normal" from button text
        self.canvas.set_title(self.fname, self.algo_type)

        # Update params for all layers to match the new algorithm type
        new_default_params = get_default_params_for_algorithm(self.algo_type)
        for layer in self.layers:
            layer["params"] = new_default_params.copy()
            # Clear intervals since they're algorithm-specific
            layer["intervals"] = {"start": [], "finish": []}

        # Update coefficients section UI
        self.coefficients_section.set_star_type(self.algo_type)
        self.coefficients_section.update_coefficients(
            self.layers[self.current_layer_idx]["params"]
        )

        # Update canvas with cleared intervals
        if len(self.JD) > 0:
            self.canvas.set_layers(self.layers)

        # Show period section only if S-DIPS is selected
        if self.algo_type == Algorithm.S_DIPS:
            self.t0_section.get_widget().setVisible(True)
            # Update error state for period field based on T0 value
            if self.T0 <= 0.0:
                self.t0_section.set_error_state(True)
            else:
                self.t0_section.set_error_state(False)
        else:
            self.t0_section.get_widget().setVisible(False)
            self.t0_section.set_error_state(False)

        self._update_interval_navigation_status()
        self._set_info(f"Star type: {self.algo_type}")

    def _on_t0_change(self, value):
        """Handle T0 value change."""
        self.T0 = value
        # Update error state: show error if S-DIPS is selected and T0 is invalid
        if self.algo_type == Algorithm.S_DIPS:
            if self.T0 <= 0.0:
                self.t0_section.set_error_state(True)
            else:
                self.t0_section.set_error_state(False)

    def _toggle_t0_helper(self):
        """Toggle T0 helper mode."""
        if len(self.JD) == 0:
            self._set_info("Please load a file first.")
            self.t0_section.btn_t0_helper.setChecked(False)
            return

        self.canvas.t0_helper_active = self.t0_section.btn_t0_helper.isChecked()

        if self.canvas.t0_helper_active:
            self._set_info(
                "T0 Helper Active!\n\n"
                "Click 2 points on the plot\n"
                "to calculate period T0.\n\n"
                "Drag points to adjust.\n"
                "Press Escape to exit."
            )
        else:
            self.canvas.clear_t0_helpers()
            self._set_info("T0 Helper deactivated.")

    def _cancel_t0_helper(self):
        """Cancel T0 helper mode."""
        if self.t0_section.btn_t0_helper.isChecked():
            self.t0_section.btn_t0_helper.setChecked(False)
            self.canvas.t0_helper_active = False
            self.canvas.clear_t0_helpers()
            self._set_info("T0 Helper cancelled.")

    def _cancel_all_modes(self):
        """Cancel all active modes (T0 helper)."""
        if self.t0_section.btn_t0_helper.isChecked():
            self._cancel_t0_helper()

    def _update_t0_from_helper(self):
        """Update T0 value from helper points."""
        t0_val = self.canvas.get_t0_value()
        if t0_val is not None:
            self.T0 = t0_val
            self.t0_section.spin_t0.setValue(t0_val)
            self._set_info(f"T0 = {t0_val:.6f} days\n({t0_val * 24:.4f} hours)")

    def _load_and_compute(self):
        """Load data and compute intervals."""
        if not self.fname and len(self.JD) == 0:
            self._set_info("Please select a file first.")
            return

        if self.fname and os.path.exists(self.fname):
            try:
                JD, mag = get_data(self.fname, -1000, 1000)
                if len(JD) == 0:
                    self._set_info("File loaded, but no valid data points.")
                    return
                self.JD, self.mag = JD, mag
            except Exception as e:
                self._set_info(f"Failed to read file: {e}")
                return

        # Disable T0 helper if active
        if self.t0_section.btn_t0_helper.isChecked():
            self._cancel_t0_helper()

        # Validate T0 for S-DIPS algorithm
        if self.algo_type == Algorithm.S_DIPS:
            if self.T0 <= 0.0:
                self.t0_section.set_error_state(True)
                self._set_info(
                    "Error: Period (T0) is required for S-DIPS algorithm.\n\n"
                    "Please enter a valid period value before computing."
                )
                self.status_bar.showMessage("Error: Period required for S-DIPS")
                return
            else:
                self.t0_section.set_error_state(False)

        # Start synchronous computation for active layer
        try:
            # Disable button during computation
            self.actions_section.btn_compute.setEnabled(False)
            self._set_info("Computing intervals...\n(This may take a moment)")
            self.status_bar.showMessage("Computing intervals...")

            # Run computation
            l_start, l_finish = self._compute_intervals_for_active_layer()

            # Re-enable button
            self.actions_section.btn_compute.setEnabled(True)

            count = len(l_start)
            total = sum(len(_["intervals"]["start"]) for _ in self.layers)
            self._update_interval_navigation_status()
            self._set_info(
                f"Layer {self.current_layer_idx + 1} updated.\n"
                f"Found {count} intervals.\n\n"
                f"Total intervals across all layers: {total}"
            )
            self.status_bar.showMessage(f"Ready - Found {count} intervals")

        except Exception as e:
            self._set_info(f"Computation failed: {e}")
            print(e, traceback.format_exc())
            self.status_bar.showMessage("Computation failed")
            self.actions_section.btn_compute.setEnabled(True)

    def _compute_intervals_for_active_layer(self):
        """Compute intervals for the currently active layer."""
        active_layer = self.layers[self.current_layer_idx]
        params = active_layer["params"]

        # Ensure params have all required keys for the current algorithm type
        # This is a safety check in case params get out of sync
        required_params = get_default_params_for_algorithm(self.algo_type)
        for key, default_value in required_params.items():
            if key not in params:
                params[key] = default_value

        # Collect excluded indices from all previous layers
        excluded_indices = set()
        for layer_idx in range(self.current_layer_idx):
            prev_layer = self.layers[layer_idx]
            prev_starts = prev_layer["intervals"]["start"]
            prev_finishes = prev_layer["intervals"]["finish"]
            for s, f in zip(prev_starts, prev_finishes):
                excluded_indices.update(range(s, f + 1))

        # Run computation synchronously
        if self.algo_type == Algorithm.GB_AT:
            l_start, l_finish = splitting_algol_configurable(
                self.JD,
                self.mag,
                index_gap=params["algol_index_gap"],
                cut_ratio=params["cut_ratio"],
                min_interval_points=params["min_interval_points"],
                excluded_indices=excluded_indices,
                fill_remaining=params.get("fill_remaining", False),
            )
        elif self.algo_type == Algorithm.MAGNITUDE_INVERTED_GB_AT:
            l_start, l_finish = splitting_algol_configurable(
                self.JD,
                self.mag,
                index_gap=params["algol_index_gap"],
                cut_ratio=params["cut_ratio"],
                min_interval_points=params["min_interval_points"],
                excluded_indices=excluded_indices,
                is_inverted=True,
                fill_remaining=params.get("fill_remaining", False),
            )
        else:  # Algorithm.S_DIPS
            l_start, l_finish = splitting_normal(
                self.JD, self.mag, self.T0, excluded_indices=excluded_indices
            )
            l_start, l_finish = check_up(self.JD, self.mag, l_start, l_finish, self.T0)

        # Update active layer results
        active_layer["intervals"]["start"] = list(l_start)
        active_layer["intervals"]["finish"] = list(l_finish)

        # Clear canvas selection
        self.canvas.clear_selection()

        # Update canvas with title
        self.canvas.set_title(self.fname, self.algo_type)
        self.canvas.set_data(
            self.JD, self.mag, reset_zoom=False
        )  # Preserve zoom when recomputing intervals
        self.canvas.set_layers(self.layers)

        # Ensure time expansion section is enabled when we have data
        self.zoom_section.set_enabled(True)

        return l_start, l_finish

    def _on_point_clicked(self):
        self._update_interval_navigation_status()

        if self.canvas.selected_interval is None and len(self.JD) > 0:
            layer = self.layers[self.current_layer_idx]
            count = len(layer["intervals"]["start"])
            total = sum(len(_["intervals"]["start"]) for _ in self.layers)
            self._set_info(
                f"Layer {self.current_layer_idx + 1} updated.\n"
                f"Found {count} intervals.\n\n"
                f"Total intervals: {total}\n"
                f"Click interval to select."
            )

    def _on_interval_selected(self, layer_idx, interval_idx):
        """Handle interval selection."""
        layer = self.layers[layer_idx]
        s = layer["intervals"]["start"][interval_idx]
        f = layer["intervals"]["finish"][interval_idx]

        # Automatically show borders when interval is selected
        self.canvas._show_resize_borders()

        # Update navigation status
        self._update_interval_navigation_status()

        self._set_info(
            f"Selected Interval\n"
            f"Layer: {layer_idx + 1}, ID: #{interval_idx}\n"
            f"Points: {s} → {f}\n\n"
            f"Press 'D' to delete.\n"
            f"Drag borders to resize."
        )

    def _remove_selected(self):
        """Remove the selected interval."""
        if self.canvas.selected_interval is None:
            self._set_info("No interval selected.\nClick an interval first.")
            return

        lay_idx, int_idx = self.canvas.selected_interval

        try:
            layer = self.layers[lay_idx]
            del layer["intervals"]["start"][int_idx]
            del layer["intervals"]["finish"][int_idx]

            self.canvas.selected_interval = None
            self.canvas.set_layers(self.layers)
            self.canvas._hide_resize_borders()
            self._update_interval_navigation_status()

            self._set_info(f"Removed interval from Layer {lay_idx + 1}.")
            self.status_bar.showMessage("Interval removed")

        except Exception as e:
            self._set_info(f"Error removing: {e}")

    def _save_intervals(self):
        """Save intervals to file."""
        if not self.fname:
            self._set_info("No file loaded.")
            return
        if len(self.JD) == 0:
            self._set_info("No data loaded.")
            return

        # Aggregate all layers
        all_start = []
        all_finish = []
        for layer in self.layers:
            all_start.extend(layer["intervals"]["start"])
            all_finish.extend(layer["intervals"]["finish"])

        if len(all_start) == 0:
            self._set_info("No intervals to save.")
            return

        # Sort by start time
        combined = sorted(zip(all_start, all_finish))
        all_start = [x[0] for x in combined]
        all_finish = [x[1] for x in combined]

        # Open save dialog
        base_name = os.path.splitext(os.path.basename(self.fname))[0]
        default_name = f"{base_name}.da!"
        suggested_path = os.path.join(os.path.dirname(self.fname), default_name)

        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Save Intervals",
            suggested_path,
            "Interval files (*.da!);;All files (*.*)",
        )

        if not fname:
            return  # User cancelled

        try:
            # Temporarily set self.fname to the chosen save location for save_data function
            original_fname = self.fname
            self.fname = fname
            out = save_data(all_start, all_finish, self.JD, self.fname)
            self.fname = original_fname  # Restore original

            self._set_info(
                f"Saved!\n{os.path.basename(out)}\nTotal intervals: {len(all_start)}"
            )
            self.status_bar.showMessage(f"Saved to {out}")
        except Exception as e:
            self._set_info(f"Save failed: {e}")

    def _reset_view(self):
        """Reset the view to show all data."""
        if len(self.JD) == 0:
            self._set_info("No data loaded.")
            return
        self.canvas.reset_view()
        self.status_bar.showMessage("View reset")

    def _update_layer_ui(self):
        """Update layer status and coefficient UI."""
        self.layers_section.update_layer_status(
            self.current_layer_idx, len(self.layers)
        )
        self.coefficients_section.update_coefficients(
            self.layers[self.current_layer_idx]["params"]
        )

    def _layer_add(self):
        """Add a new layer."""
        current_layer = self.layers[self.current_layer_idx]
        new_layer = {
            "params": current_layer["params"].copy(),
            "intervals": {"start": [], "finish": []},
        }
        self.layers.append(new_layer)
        self.current_layer_idx = len(self.layers) - 1
        self._update_layer_ui()
        self._update_interval_navigation_status()
        self._set_info(
            f"Added Layer {len(self.layers)}.\nParams copied, empty intervals."
        )

    def _layer_remove(self):
        """Remove the current layer."""
        if len(self.layers) <= 1:
            self._set_info("Cannot remove the last layer.")
            return

        del self.layers[self.current_layer_idx]
        if self.current_layer_idx >= len(self.layers):
            self.current_layer_idx = len(self.layers) - 1

        self._update_layer_ui()
        self._update_interval_navigation_status()
        self.canvas.set_layers(self.layers)

        # Clear selection since the layer was removed
        if self.canvas.selected_interval is not None:
            self.canvas.selected_interval = None
            self.canvas._update_selection_appearance()

        self._set_info(
            f"Removed layer.\nNow showing Layer {self.current_layer_idx + 1}"
        )

    def _layer_prev(self):
        """Switch to previous layer."""
        if self.current_layer_idx > 0:
            self.current_layer_idx -= 1
            self.canvas.clear_selection()
            self._update_layer_ui()
            self._update_interval_navigation_status()
        else:
            self._set_info("Already at first layer.")

    def _layer_next(self):
        """Switch to next layer."""
        if self.current_layer_idx < len(self.layers) - 1:
            self.current_layer_idx += 1
            self.canvas.clear_selection()
            self._update_layer_ui()
            self._update_interval_navigation_status()
        else:
            self._set_info("Already at last layer.")

    def _on_coefficient_change(self, key: str, value):
        """Handle coefficient change."""
        self.layers[self.current_layer_idx]["params"][key] = value

    def _on_time_zoom_change(self, zoom_factor: float):
        """Handle time zoom factor change."""
        self.canvas.set_time_zoom(zoom_factor)

    def _on_magnitude_zoom_change(self, zoom_factor: float):
        """Handle magnitude zoom factor change."""
        self.canvas.set_magnitude_zoom(zoom_factor)

    def _on_zoom_reset(self):
        """Handle zoom reset (both time and magnitude)."""
        self.canvas.reset_time_zoom()
        self.canvas.reset_magnitude_zoom()

    def _select_interval_by_index(self, interval_idx):
        """Programmatically select an interval by its index in the current layer."""
        layer = self.layers[self.current_layer_idx]
        if 0 <= interval_idx < len(layer["intervals"]["start"]):
            # Clear any existing selection first
            self.canvas.clear_selection()
            # Set the new selection
            self.canvas.selected_interval = (self.current_layer_idx, interval_idx)
            # Update appearance
            self.canvas._update_selection_appearance()
            # Show borders
            self.canvas._show_resize_borders()
            # Emit the selection signal
            self.canvas.interval_selected.emit(self.current_layer_idx, interval_idx)
            return True
        return False

    def _update_interval_navigation_status(self):
        """Update the interval navigation status display."""
        layer = self.layers[self.current_layer_idx]
        total_intervals = len(layer["intervals"]["start"])

        current_idx = -1
        if self.canvas.selected_interval is not None:
            sel_layer_idx, sel_int_idx = self.canvas.selected_interval
            if sel_layer_idx == self.current_layer_idx:
                current_idx = sel_int_idx

        self.interval_nav_section.update_interval_status(total_intervals, current_idx)

    def _interval_prev(self):
        """Navigate to previous interval."""
        layer = self.layers[self.current_layer_idx]
        total_intervals = len(layer["intervals"]["start"])

        if total_intervals == 0:
            return

        current_idx = -1
        if self.canvas.selected_interval is not None:
            sel_layer_idx, sel_int_idx = self.canvas.selected_interval
            if sel_layer_idx == self.current_layer_idx:
                current_idx = sel_int_idx

        # If no selection or at first interval, select the last interval
        if current_idx <= 0:
            target_idx = total_intervals - 1
        else:
            target_idx = current_idx - 1

        self._select_interval_by_index(target_idx)
        self._update_interval_navigation_status()

    def _interval_next(self):
        """Navigate to next interval."""
        layer = self.layers[self.current_layer_idx]
        total_intervals = len(layer["intervals"]["start"])

        if total_intervals == 0:
            return

        current_idx = -1
        if self.canvas.selected_interval is not None:
            sel_layer_idx, sel_int_idx = self.canvas.selected_interval
            if sel_layer_idx == self.current_layer_idx:
                current_idx = sel_int_idx

        # If no selection or at last interval, select the first interval
        if current_idx >= total_intervals - 1 or current_idx == -1:
            target_idx = 0
        else:
            target_idx = current_idx + 1

        self._select_interval_by_index(target_idx)
        self._update_interval_navigation_status()
