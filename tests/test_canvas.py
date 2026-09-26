"""Self-check of the light-curve canvas axes (needs a display). Run: xvfb-run -a uv run python tests/test_canvas.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from src.canvas import LightcurveCanvas  # noqa: E402


def test_axis_labels_follow_a_resize():
    w = LightcurveCanvas()
    w.resize(900, 500)
    w.show()
    t = np.linspace(1325, 1350, 3000)
    w.set_data(t, 40 * np.exp(-(((t - 1337) / 0.1) ** 2)))
    w.resize(1500, 800)  # a maximised window: the labels were left over the stretched axis
    for _ in range(10):
        app.processEvents()
    axes = [c for c in w.canvas.central_widget.children[0].children if hasattr(c, "_axis_ends")]
    assert len(axes) == 2
    for ax in axes:
        k = 1 if ax.orientation == "left" else 0
        shown = ax.node_transform(w.view.scene).map(ax._axis_ends())[:, k]
        assert np.allclose(ax.axis.domain, shown), (ax.orientation, ax.axis.domain, shown)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
