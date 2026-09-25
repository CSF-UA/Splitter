"""VisPy light curve: y = -mag (brighter is up), one shaded band per interval."""

import os

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget
from vispy import scene
from vispy.color import Color
from vispy.scene import AxisWidget, visuals

from src.constants import COLORS, INTERVAL_COLORS


class LightcurveCanvas(QWidget):
    interval_selected = Signal(int, int)  # layer, interval
    background_clicked = Signal()
    period_measured = Signal(float)

    def __init__(self):
        super().__init__()
        self.canvas = scene.SceneCanvas(keys="interactive", bgcolor=COLORS["plot_bg"], parent=self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas.native)

        grid = self.canvas.central_widget.add_grid(spacing=0)
        self.title = scene.Label("", color=COLORS["text"], font_size=12, bold=True)
        grid.add_widget(self.title, row=0, col=2).height_max = 30
        style = dict(text_color=COLORS["text"], axis_color=COLORS["text_dim"],
                     tick_color=COLORS["text_dim"], axis_font_size=9, tick_font_size=8)
        yaxis = AxisWidget(orientation="left", axis_label="magnitude, mmag", **style)
        xaxis = AxisWidget(orientation="bottom", axis_label="JD - 2 457 000", **style)
        yaxis.width_max, xaxis.height_max = 60, 40
        grid.add_widget(row=1, col=0).width_max = 25
        grid.add_widget(yaxis, row=1, col=1)
        self.view = grid.add_view(row=1, col=2, camera="panzoom")
        self.view.camera.aspect = None
        grid.add_widget(xaxis, row=2, col=2)
        grid.add_widget(row=3, col=2).height_max = 25
        xaxis.link_view(self.view)
        yaxis.link_view(self.view)

        # created in drawing order: bands under points, borders and period helper on top
        self.bands = visuals.Mesh(parent=self.view.scene)
        self.points = visuals.Markers(parent=self.view.scene)
        self.points.set_gl_state(depth_test=False)
        self.borders = visuals.Line(color=COLORS["accent"], width=2, connect="segments", parent=self.view.scene)
        self.borders.set_gl_state(depth_test=False)
        self.helper_line = visuals.Line(color=COLORS["t0_marker"], width=2, parent=self.view.scene)
        self.helper = visuals.Markers(parent=self.view.scene)
        for v in (self.bands, self.points, self.borders, self.helper_line, self.helper):
            v.visible = False

        self.JD = self.mag = np.array([])
        self.layers = []
        self.selected = None  # (layer, interval)
        self.x_zoom = self.y_zoom = 1.0
        self.helper_active = False
        self.helper_points = []
        self._drag = None  # "left" / "right" border, or the index of a helper point
        self.canvas.events.mouse_press.connect(self._press)
        self.canvas.events.mouse_move.connect(self._move)
        self.canvas.events.mouse_release.connect(self._release)

    def set_title(self, fname, algo):
        self.title.text = f"{os.path.basename(fname)} | {algo}" if fname else ""

    def set_data(self, JD, mag, reset_zoom=True):
        changed = not (np.array_equal(self.JD, JD) and np.array_equal(self.mag, mag))
        self.JD, self.mag = JD, mag
        if reset_zoom or changed:
            self.x_zoom = self.y_zoom = 1.0
        self._update_camera()

    def set_layers(self, layers):
        self.layers = layers
        self.redraw()

    def select(self, key):
        self.selected = key
        self.redraw()
        if key is not None:
            self.interval_selected.emit(*key)

    def set_zoom(self, x=None, y=None):
        self.x_zoom = x or self.x_zoom
        self.y_zoom = y or self.y_zoom
        self._update_camera()

    def _update_camera(self):
        if len(self.JD) == 0:
            return
        lo, hi = self.JD.min(), self.JD.max()
        cx, wx = (lo + hi) / 2, (hi - lo) * 1.04 / self.x_zoom  # 2 % margin on each side
        m0, m1 = self.mag.min(), self.mag.max()
        cy, hy = -(m0 + m1) / 2, (m1 - m0) * 1.2 / self.y_zoom  # 10 % margin on each side
        # z given too: otherwise VisPy asks every visual (the empty helper markers too) for scene bounds
        self.view.camera.set_range(x=(cx - wx / 2, cx + wx / 2), y=(cy - hy / 2, cy + hy / 2), z=(0, 0))

    def _y_span(self):
        pad = (self.mag.max() - self.mag.min()) * 0.1
        return -self.mag.max() - pad, -self.mag.min() + pad

    def _intervals(self):
        for li, layer in enumerate(self.layers):
            iv = layer["intervals"]
            for i, (s, f) in enumerate(zip(iv["start"], iv["finish"])):
                yield li, i, s, f

    def redraw(self):
        """Rebuild point colours, interval bands and the selection borders."""
        if len(self.JD) == 0:
            for v in (self.bands, self.points, self.borders):
                v.visible = False
            self.canvas.update()
            return
        colors = np.tile(Color(COLORS["point_data"]).rgba, (len(self.JD), 1)).astype(np.float32)
        y0, y1 = self._y_span()
        verts, band_colors = [], []
        for li, i, s, f in self._intervals():
            c = Color(INTERVAL_COLORS[(7 * li + i) % len(INTERVAL_COLORS)])
            colors[s : f + 1] = c.rgba
            c.alpha = 0.4 if self.selected == (li, i) else 0.15
            verts += [[self.JD[s], y0, 0], [self.JD[f], y0, 0], [self.JD[f], y1, 0], [self.JD[s], y1, 0]]
            band_colors += [c.rgba] * 4
        self.points.set_data(np.c_[self.JD, -self.mag], face_color=colors, edge_color=None, size=6)
        self.points.visible = True
        if verts:
            q = np.arange(len(verts) // 4)[:, None] * 4
            self.bands.set_data(vertices=np.array(verts, np.float32),
                                faces=np.vstack([q + [0, 1, 2], q + [0, 2, 3]]).astype(np.uint32),
                                vertex_colors=np.array(band_colors, np.float32))
        self.bands.visible = bool(verts)
        self._update_borders()
        self.canvas.update()

    def _update_borders(self):
        self.borders.visible = self.selected is not None
        if self.selected is not None:
            iv = self.layers[self.selected[0]]["intervals"]
            xs, xf = self.JD[iv["start"][self.selected[1]]], self.JD[iv["finish"][self.selected[1]]]
            y0, y1 = self._y_span()
            self.borders.set_data(pos=np.array([[xs, y0], [xs, y1], [xf, y0], [xf, y1]]))

    def _update_helper(self):
        pts = np.array([[px, -py] for px, py in self.helper_points]).reshape(-1, 2)
        self.helper.visible = len(pts) > 0
        self.helper_line.visible = len(pts) == 2
        if len(pts):
            self.helper.set_data(pts, face_color=COLORS["t0_marker"], edge_color="white", size=15, edge_width=2)
        if len(pts) == 2:
            self.helper_line.set_data(pos=pts)
            self.period_measured.emit(float(abs(pts[1, 0] - pts[0, 0])))
        self.canvas.update()

    def clear_helper(self):
        self.helper_points = []
        self._update_helper()

    def _to_data(self, pos):
        x, y = self.canvas.scene.node_transform(self.view.scene).map(pos)[:2]
        return x, -y  # back to magnitudes

    def _near_border(self, x):
        if self.selected is None:
            return None
        iv = self.layers[self.selected[0]]["intervals"]
        xs, xf = self.JD[iv["start"][self.selected[1]]], self.JD[iv["finish"][self.selected[1]]]
        tol = max((xf - xs) * 0.1, 0.01)
        return "left" if abs(x - xs) < tol else "right" if abs(x - xf) < tol else None

    def _press(self, event):
        if event.button != 1 or len(self.JD) == 0:
            return
        x, y = self._to_data(event.pos)
        if self.helper_active:
            near = [k for k, (px, py) in enumerate(self.helper_points) if abs(x - px) < 0.5 and abs(y - py) < 0.02]
            if near:
                self._drag = near[0]
            else:
                self.helper_points = (self.helper_points + [(x, y)])[-2:]
                self._update_helper()
            return
        self._drag = self._near_border(x)
        if self._drag:
            self.view.camera.interactive = False
            return
        y0, y1 = self._y_span()
        for li, i, s, f in self._intervals():
            if self.JD[s] <= x <= self.JD[f] and y0 <= -y <= y1:
                return self.select((li, i))
        self.select(None)
        self.background_clicked.emit()

    def _move(self, event):
        if len(self.JD) == 0:
            return
        x, y = self._to_data(event.pos)
        if isinstance(self._drag, int):
            self.helper_points[self._drag] = (x, y)
            self._update_helper()
        elif self._drag:
            iv = self.layers[self.selected[0]]["intervals"]
            i = self.selected[1]
            k = int(np.clip(np.searchsorted(self.JD, x), 0, len(self.JD) - 1))
            if k > 0 and abs(self.JD[k - 1] - x) < abs(self.JD[k] - x):
                k -= 1
            if self._drag == "left":
                iv["start"][i] = max(0, min(k, iv["finish"][i] - 1))
            else:
                iv["finish"][i] = min(len(self.JD) - 1, max(k, iv["start"][i] + 1))
            self._update_borders()  # full redraw on release keeps dragging fast
            self.canvas.update()
        elif self.selected is not None:
            self.canvas.native.setCursor(Qt.SizeHorCursor if self._near_border(x) else Qt.ArrowCursor)

    def _release(self, event):
        if self._drag in ("left", "right"):
            self.view.camera.interactive = True
            self.redraw()
        self._drag = None
