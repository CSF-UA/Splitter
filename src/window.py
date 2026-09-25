"""Splitter main window: controls on both sides, the VisPy light curve in the middle."""

import os
import traceback

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QRadioButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from src.auto import auto_split
from src.canvas import LightcurveCanvas
from src.constants import COLORS
from src.core import check_up, get_data, save_data, splitting_algol_configurable, splitting_normal

# per algorithm: (key, label, default, min, max, step); the widget follows the default's type
COEFFS = {
    "Auto": [
        ("frac", "Frac (0=auto):", 0.0, 0.0, 0.99, 0.05),
        ("min_points", "Min Points:", 15, 3, 1000, 1),
        ("minima_only", "Minima only:", False, 0, 0, 0),
    ],
    "GB-AT": [
        ("algol_index_gap", "Index Gap:", 2, 1, 1000, 1),
        ("cut_ratio", "Cut Ratio:", 0.25, 0.01, 0.99, 0.05),
        ("min_interval_points", "Min Points:", 5, 1, 1000, 1),
        ("fill_remaining", "Fill Remaining:", False, 0, 0, 0),
    ],
    "M-inverted GB-AT": [
        ("algol_index_gap", "Index Gap:", 2, 1, 100, 1),
        ("cut_ratio", "Cut Ratio:", 0.25, 0.01, 0.99, 0.05),
        ("min_interval_points", "Min Points:", 5, 1, 100, 1),
        ("fill_remaining", "Fill Remaining:", False, 0, 0, 0),
    ],
    "S-DIPS": [("alpha", "Alpha:", 0.12, 0.01, 100.0, 0.01)],
}

STYLE = f"""
    QMainWindow {{ background-color: {COLORS["bg_dark"]}; }}
    QWidget {{ color: {COLORS["text"]}; font-family: 'Segoe UI', 'SF Pro Display', sans-serif; font-size: 12px; }}
    QGroupBox {{ background-color: {COLORS["bg_panel"]}; border: 1px solid {COLORS["border"]}; border-radius: 8px;
                 margin-top: 12px; padding-top: 8px; font-weight: bold; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; color: {COLORS["accent"]}; }}
    QPushButton {{ background-color: {COLORS["bg_button"]}; border: 1px solid {COLORS["border"]}; border-radius: 6px;
                   padding: 6px 12px; color: {COLORS["text"]}; }}
    QPushButton:hover {{ background-color: {COLORS["bg_button_hover"]}; }}
    QPushButton:pressed {{ background-color: {COLORS["accent"]}; color: white; }}
    QPushButton:checked {{ background-color: {COLORS["accent"]}; color: white; }}
    QLineEdit, QSpinBox, QDoubleSpinBox {{ background-color: {COLORS["bg_panel"]}; border: 1px solid {COLORS["border"]};
                                           border-radius: 4px; padding: 4px 8px; color: {COLORS["text"]}; }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {COLORS["accent"]}; }}
    QRadioButton {{ spacing: 8px; color: {COLORS["text"]}; }}
    QRadioButton::indicator {{ width: 16px; height: 16px; border-radius: 8px; border: 2px solid {COLORS["border"]};
                               background-color: {COLORS["bg_panel"]}; }}
    QRadioButton::indicator:checked {{ background-color: {COLORS["accent"]}; border-color: {COLORS["accent"]}; }}
    QStatusBar {{ background-color: {COLORS["bg_panel"]}; color: {COLORS["text_dim"]}; border-top: 1px solid {COLORS["border"]}; }}
    QLabel {{ background-color: transparent; }}
"""


def defaults(algo):
    return {key: default for key, _, default, *_ in COEFFS[algo]}


def new_layer(params):
    return {"params": params, "intervals": {"start": [], "finish": [], "kind": []}}


def button(text, slot, width=None):
    b = QPushButton(text)
    b.clicked.connect(slot)
    if width:
        b.setFixedWidth(width)
    return b


def row(*widgets):
    """Horizontal layout; None adds a stretch."""
    lay = QHBoxLayout()
    for w in widgets:
        if w is None:
            lay.addStretch()
        else:
            lay.addWidget(w)
    return lay


def group(title, *items):
    """Group box with a vertical layout of widgets and layouts."""
    box = QGroupBox(title)
    lay = QVBoxLayout(box)
    for item in items:
        if isinstance(item, QWidget):
            lay.addWidget(item)
        else:
            lay.addLayout(item)
    return box


def panel(widgets, width):
    w = QWidget()
    w.setFixedWidth(width)
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(12)
    for x in widgets:
        lay.addWidget(x)
    lay.addStretch()
    return w


class SplitterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Splitter v5.0.0")
        self.setMinimumSize(1400, 800)
        self.setStyleSheet(STYLE)
        self.fname, self.algo, self.P = "", "Auto", 0.0
        self.JD = self.mag = np.array([])
        self.layers, self.cur = [new_layer(defaults(self.algo))], 0

        self.canvas = LightcurveCanvas()
        self.canvas.interval_selected.connect(self._on_selected)
        self.canvas.background_clicked.connect(self._on_background)
        self.canvas.period_measured.connect(self._on_period_measured)

        # left panel
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        self.algo_group = QButtonGroup(self)
        radios = []
        for name in COEFFS:
            rb = QRadioButton(name)
            rb.setChecked(name == self.algo)
            self.algo_group.addButton(rb)
            radios.append(rb)
        self.algo_group.buttonClicked.connect(self._on_algo)
        self.period = QDoubleSpinBox()
        self.period.setRange(0.0, 1000.0)
        self.period.setDecimals(6)
        self.period.setSingleStep(0.1)
        self.period.valueChanged.connect(self._on_period)
        self.helper_btn = QPushButton("P [+]")
        self.helper_btn.setCheckable(True)
        self.helper_btn.toggled.connect(self._toggle_helper)
        self.period_box = group("Period P, d", row(self.period, self.helper_btn))
        self.compute_btn = button("Load + Compute", self._compute)
        self.compute_btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLORS['success']}; color: white; font-weight: bold; padding: 8px; }}"
            "QPushButton:hover { background-color: #22c55e; }"
        )
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setMinimumHeight(100)
        self.info.setStyleSheet(
            f"background-color: {COLORS['bg_panel']}; border: 1px solid {COLORS['border']}; border-radius: 6px;"
            f"padding: 10px; color: {COLORS['text']}; font-size: 11px;"
        )
        left = [
            group("File", button("Open File...", self._browse), self.file_label),
            group("Algorithm", *radios),
            self.period_box,
            group("Actions", self.compute_btn, button("Remove Selected", self._remove),
                  button("Save (.txt file)", self._save), button("Reset View", self._reset_view)),
            self.info,
        ]

        # right panel
        self.layer_label, self.nav_label = QLabel(), QLabel()
        for label in (self.layer_label, self.nav_label):
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-weight: bold; font-size: 12px;")
        plus, minus = button("+", self._layer_add, 40), button("−", self._layer_remove, 40)
        for b in (plus, minus):
            b.setStyleSheet("font-weight: bold;")
        self.nav_buttons = [button("◀", lambda: self._nav(-1), 40), button("▶", lambda: self._nav(1), 40)]
        self.coeff_form = QFormLayout()
        coeff_box = QGroupBox("Coefficients")
        coeff_box.setLayout(self.coeff_form)
        self.zoom_setters = {}
        self.zoom_box = group("Zoom", self._zoom_controls("x"), self._zoom_controls("y"))
        self.zoom_box.setEnabled(False)
        right = [
            group("Layers", self.layer_label, row(button("◀", lambda: self._layer_step(-1), 40),
                                                  button("▶", lambda: self._layer_step(1), 40), None, plus, minus)),
            group("Interval Navigation", self.nav_label, row(*self.nav_buttons)),
            coeff_box,
            self.zoom_box,
        ]

        central = QWidget()
        main = QHBoxLayout(central)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(8)
        main.addWidget(panel(left, 220))
        main.addWidget(self.canvas, stretch=1)
        main.addWidget(panel(right, 200))
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

        for key, slot in (("D", self._remove), ("S", self._save), ("R", self._reset_view), ("Escape", self._cancel_helper)):
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(slot)
            self.addAction(action)

        self._refresh_layer()
        self._refresh_nav()
        self._info("Welcome!\n\n1. Open a .tess file\n2. Adjust P if needed\n3. Click 'Load + Compute'\n\n"
                   "Shortcuts: D=delete, S=save, R=reset")

    # ---- widgets ------------------------------------------------------------
    def _zoom_controls(self, axis):
        spin = QDoubleSpinBox()
        spin.setRange(0.1, 100.0)
        spin.setDecimals(2)
        spin.setReadOnly(True)
        spin.setValue(1.0)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(1, 1000)
        slider.setValue(10)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(100)

        def set_zoom(value, from_slider=False):
            value = min(100.0, max(0.1, value))
            spin.setValue(value)
            if not from_slider:
                slider.blockSignals(True)
                slider.setValue(int(1 + (value - 0.1) / 99.9 * 999))
                slider.blockSignals(False)
            self.canvas.set_zoom(**{axis: value})

        slider.valueChanged.connect(lambda s: set_zoom(0.1 + (s - 1) / 999 * 99.9, True))
        self.zoom_setters[axis] = set_zoom
        lay = QVBoxLayout()
        lay.addLayout(row(QLabel(f"{axis.upper()}:"), spin))
        lay.addWidget(slider)
        lay.addLayout(row(None, button("+", lambda: set_zoom(spin.value() + 0.1), 40),
                          button("-", lambda: set_zoom(spin.value() - 0.1), 40),
                          button("⟳", lambda: set_zoom(1.0), 40), None))
        return lay

    def _refresh_layer(self):
        self.layer_label.setText(f"Layer {self.cur + 1} / {len(self.layers)}")
        while self.coeff_form.rowCount():
            self.coeff_form.removeRow(0)
        params = self.layers[self.cur]["params"]
        for key, label, default, lo, hi, step in COEFFS[self.algo]:
            def store(value, key=key):
                params[key] = value

            if isinstance(default, bool):
                w = QCheckBox()
                w.setChecked(params[key])
                w.toggled.connect(store)
            else:
                w = QSpinBox() if isinstance(default, int) else QDoubleSpinBox()
                w.setRange(lo, hi)
                w.setSingleStep(step)
                w.setValue(params[key])
                w.valueChanged.connect(store)
            self.coeff_form.addRow(label, w)

    def _refresh_nav(self):
        n = len(self.layers[self.cur]["intervals"]["start"])
        sel = self.canvas.selected
        i = sel[1] if sel and sel[0] == self.cur else -1
        self.nav_label.setText("No intervals" if n == 0 else f"Interval {i + 1} / {n}" if i >= 0 else f"{n} intervals")
        for b in self.nav_buttons:
            b.setEnabled(n > 0)

    def _info(self, text, status=None):
        self.info.setText(text)
        if status:
            self.statusBar().showMessage(status)

    def _counts(self):
        return len(self.layers[self.cur]["intervals"]["start"]), sum(len(l["intervals"]["start"]) for l in self.layers)

    # ---- file and algorithm -----------------------------------------------------
    def _browse(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select .tess file", "", "TESS/LC files (*.tess);;All files (*.*)")
        if fname:
            self.fname = fname
            self.file_label.setText(os.path.basename(fname))
            self._load_preview()

    def _load_preview(self):
        try:
            JD, mag = get_data(self.fname)
        except Exception as e:
            return self._info(f"Preview failed: {e}")
        if len(JD) == 0:
            return self._info("File loaded, but no valid data points.")
        self.JD, self.mag = JD, mag
        self.layers, self.cur = [new_layer(defaults(self.algo))], 0
        self.period.setValue(0.0)
        self.canvas.selected = None
        self.canvas.set_title(self.fname, self.algo)
        self.canvas.set_data(JD, mag)
        self.canvas.set_layers(self.layers)
        for set_zoom in self.zoom_setters.values():
            set_zoom(1.0)
        self.zoom_box.setEnabled(True)
        self._refresh_layer()
        self._refresh_nav()
        self._info(f"Loaded {len(JD)} points\n\nAdjust P and coefficients,\nthen click 'Load + Compute'\nto detect intervals.",
                   f"Loaded: {os.path.basename(self.fname)}")

    def _on_algo(self, btn):
        self.algo = btn.text()
        for layer in self.layers:
            layer["params"] = defaults(self.algo)
        self.canvas.set_title(self.fname, self.algo)
        self.period_box.setVisible(self.algo in ("Auto", "S-DIPS"))
        self._on_period(self.P)
        self._refresh_layer()
        self._refresh_nav()
        self._info(f"Algorithm: {self.algo}")

    def _on_period(self, value):
        self.P = value
        bad = self.algo == "S-DIPS" and value <= 0
        self.period.setStyleSheet(f"QDoubleSpinBox {{ border: 2px solid {COLORS['t0_marker']}; }}" if bad else "")

    def _toggle_helper(self, on):
        if on and len(self.JD) == 0:
            self.helper_btn.setChecked(False)
            return self._info("Please load a file first.")
        self.canvas.helper_active = on
        if on:
            self._info("P Helper Active!\n\nClick 2 points on the plot\nto calculate period P.\n\n"
                       "Drag points to adjust.\nPress Escape to exit.")
        else:
            self.canvas.clear_helper()
            self._info("P Helper deactivated.")

    def _cancel_helper(self):
        if self.helper_btn.isChecked():
            self.helper_btn.setChecked(False)

    def _on_period_measured(self, p):
        self.period.setValue(p)
        self._info(f"P = {p:.6f} days\n({p * 24:.4f} hours)")

    # ---- computing ----------------------------------------------------------
    def _compute(self):
        if not self.fname and len(self.JD) == 0:
            return self._info("Please select a file first.")
        if self.fname and os.path.exists(self.fname):
            try:
                JD, mag = get_data(self.fname)
            except Exception as e:
                return self._info(f"Failed to read file: {e}")
            if len(JD) == 0:
                return self._info("File loaded, but no valid data points.")
            if not (np.array_equal(JD, self.JD) and np.array_equal(mag, self.mag)):
                for layer in self.layers:  # the file changed on disk: old indices point at other points
                    layer["intervals"] = {"start": [], "finish": [], "kind": []}
                self.canvas.selected = None
            self.JD, self.mag = JD, mag
        self._cancel_helper()
        if self.algo == "S-DIPS" and self.P <= 0:
            self._on_period(self.P)
            return self._info("Error: Period (P) is required for S-DIPS algorithm.\n\n"
                              "Please enter a valid period value before computing.", "Error: Period required for S-DIPS")
        self.compute_btn.setEnabled(False)
        self._info("Computing intervals...\n(This may take a moment)", "Computing intervals...")
        try:
            extra = self._run_algorithm()
        except Exception as e:
            traceback.print_exc()
            return self._info(f"Computation failed: {e}", "Computation failed")
        finally:
            self.compute_btn.setEnabled(True)
        n, total = self._counts()
        self._info(f"Layer {self.cur + 1} updated.\nFound {n} intervals.\n\nTotal intervals across all layers: {total}{extra}",
                   f"Ready - Found {n} intervals")

    def _run_algorithm(self):
        layer = self.layers[self.cur]
        p = layer["params"]
        excluded = {k for l in self.layers[: self.cur]
                    for s, f in zip(l["intervals"]["start"], l["intervals"]["finish"]) for k in range(s, f + 1)}
        extra = ""
        if self.algo == "Auto":
            s, f, kinds, info = auto_split(self.JD, self.mag, self.P, p["frac"], p["min_points"], p["minima_only"], excluded)
            if info["period"] > 0:
                self.period.setValue(info["period"])
                extra = f"\n\nP = {info['period']:.6f} d, type {info['type']}\nRejected: {info['rejected'] or 'none'}"
            else:
                extra = "\n\nNo reliable period found.\nEnter P manually and compute again."
        elif self.algo == "S-DIPS":
            s, f, kinds = check_up(self.JD, self.mag, *splitting_normal(self.JD, self.mag, self.P, p["alpha"], excluded), self.P)
        else:
            s, f, kinds = splitting_algol_configurable(
                self.JD, self.mag, p["algol_index_gap"], p["cut_ratio"], p["min_interval_points"], excluded,
                self.algo == "M-inverted GB-AT", p["fill_remaining"])
        layer["intervals"] = {"start": list(s), "finish": list(f), "kind": list(kinds)}
        self.canvas.selected = None
        self.canvas.set_title(self.fname, self.algo)
        self.canvas.set_data(self.JD, self.mag, reset_zoom=False)
        self.canvas.set_layers(self.layers)
        self.zoom_box.setEnabled(True)
        self._refresh_nav()
        return extra

    # ---- intervals ------------------------------------------------------------
    def _on_selected(self, li, i):
        iv = self.layers[li]["intervals"]
        self._refresh_nav()
        self._info(f"Selected Interval\nLayer: {li + 1}, ID: #{i} ({iv['kind'][i]})\n"
                   f"Points: {iv['start'][i]} → {iv['finish'][i]}\n\nPress 'D' to delete.\nDrag borders to resize.")

    def _on_background(self):
        self._refresh_nav()
        n, total = self._counts()
        self._info(f"Layer {self.cur + 1} updated.\nFound {n} intervals.\n\nTotal intervals: {total}\nClick interval to select.")

    def _remove(self):
        if self.canvas.selected is None:
            return self._info("No interval selected.\nClick an interval first.")
        li, i = self.canvas.selected
        for key in ("start", "finish", "kind"):
            del self.layers[li]["intervals"][key][i]
        self.canvas.selected = None
        self.canvas.redraw()
        self._refresh_nav()
        self._info(f"Removed interval from Layer {li + 1}.", "Interval removed")

    def _save(self):
        if not self.fname or len(self.JD) == 0:
            return self._info("No file loaded." if not self.fname else "No data loaded.")
        rows = sorted(t for l in self.layers
                      for t in zip(l["intervals"]["start"], l["intervals"]["finish"], l["intervals"]["kind"]))
        if not rows:
            return self._info("No intervals to save.")
        suggested = os.path.splitext(self.fname)[0] + "_intervals.txt"
        fname, _ = QFileDialog.getSaveFileName(self, "Save Intervals", suggested, "Interval .txt file (*.txt);;All files (*.*)")
        if not fname:
            return
        try:
            out = save_data(*zip(*rows), fname)
        except Exception as e:
            return self._info(f"Save failed: {e}")
        self._info(f"Saved!\n{os.path.basename(out)}\nTotal intervals: {len(rows)}", f"Saved to {out}")

    def _reset_view(self):
        if len(self.JD) == 0:
            return self._info("No data loaded.")
        for set_zoom in self.zoom_setters.values():
            set_zoom(1.0)
        self.statusBar().showMessage("View reset")

    def _nav(self, step):
        n = len(self.layers[self.cur]["intervals"]["start"])
        if n == 0:
            return
        sel = self.canvas.selected
        if sel and sel[0] == self.cur:
            self.canvas.select((self.cur, (sel[1] + step) % n))
        else:
            self.canvas.select((self.cur, n - 1 if step < 0 else 0))

    # ---- layers ---------------------------------------------------------------
    def _layer_add(self):
        self.layers.append(new_layer(dict(self.layers[self.cur]["params"])))
        self.cur = len(self.layers) - 1
        self._refresh_layer()
        self._refresh_nav()
        self._info(f"Added Layer {len(self.layers)}.\nParams copied, empty intervals.")

    def _layer_remove(self):
        if len(self.layers) <= 1:
            return self._info("Cannot remove the last layer.")
        del self.layers[self.cur]
        self.cur = min(self.cur, len(self.layers) - 1)
        self.canvas.selected = None
        self.canvas.set_layers(self.layers)
        self._refresh_layer()
        self._refresh_nav()
        self._info(f"Removed layer.\nNow showing Layer {self.cur + 1}")

    def _layer_step(self, step):
        if not 0 <= self.cur + step < len(self.layers):
            return self._info("Already at first layer." if step < 0 else "Already at last layer.")
        self.cur += step
        self.canvas.select(None)
        self._refresh_layer()
        self._refresh_nav()
