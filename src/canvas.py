import os
from PySide6.QtWidgets import QVBoxLayout, QWidget
from PySide6.QtCore import Signal
import numpy as np
from vispy import scene
from vispy.scene import visuals, AxisWidget
from vispy.color import Color

from src.models import Algorithm

from src.contants import COLORS, INTERVAL_COLORS


class LightcurveCanvas(QWidget):
    """VisPy-based lightcurve visualization widget."""

    point_clicked = Signal(float, float)  # Emitted when clicking on canvas
    interval_selected = Signal(int, int)  # layer_idx, interval_idx
    t0_updated = Signal(float)  # Emitted when T0 value is calculated from helper points

    def __init__(self, parent=None):
        super().__init__(parent)

        # White/light background like matplotlib v4.0
        self.canvas = scene.SceneCanvas(
            keys="interactive", bgcolor=COLORS["plot_bg"], parent=self
        )
        self.canvas.native.setParent(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas.native)

        # Create grid layout for axes
        self.grid = self.canvas.central_widget.add_grid(spacing=0)

        # Title at top (row 0, col 2 - above main view)
        self.title_text = scene.Label(
            text="",
            color=COLORS["text"],
            font_size=12,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )
        self.title_widget = self.grid.add_widget(self.title_text, row=0, col=2)
        self.title_widget.height_max = 30

        # Y-axis label (row 1, col 0) - rotated text for "magnitude, mmag"
        self.ylabel_widget = self.grid.add_widget(row=1, col=0)
        self.ylabel_widget.width_max = 25

        # Y-axis with ticks (row 1, col 1)
        self.yaxis = AxisWidget(
            orientation="left",
            axis_label="magnitude, mmag",
            text_color=COLORS["text"],
            axis_color=COLORS["text_dim"],
            tick_color=COLORS["text_dim"],
            axis_font_size=9,
            tick_font_size=8,
        )
        self.yaxis.width_max = 60
        self.grid.add_widget(self.yaxis, row=1, col=1)

        # Main view for plot (row 1, col 2)
        self.view = self.grid.add_view(row=1, col=2, camera="panzoom")
        self.view.camera.aspect = (
            None  # Auto aspect - don't force 1:1 for lightcurve data
        )
        self.view.camera.interactive = True  # Initially allow interaction

        # X-axis with ticks (row 2, col 2)
        self.xaxis = AxisWidget(
            orientation="bottom",
            axis_label="JD - 2 457 000",
            text_color=COLORS["text"],
            axis_color=COLORS["text_dim"],
            tick_color=COLORS["text_dim"],
            axis_font_size=9,
            tick_font_size=8,
        )
        self.xaxis.height_max = 40
        self.grid.add_widget(self.xaxis, row=2, col=2)

        # X-axis label (row 3, col 2)
        self.xlabel_widget = self.grid.add_widget(row=3, col=2)
        self.xlabel_widget.height_max = 25

        # Link axes to view
        self.xaxis.link_view(self.view)
        self.yaxis.link_view(self.view)

        # Visual elements
        self.scatter = None
        self.interval_mesh = None  # Mesh for interval rectangles
        self.interval_scatter = None  # Single scatter for all interval points
        self.interval_metadata = []  # List of (layer_idx, interval_idx, x_start, x_end)
        self.t0_markers = None
        self.t0_line = None

        # Data
        self.JD = np.array([])
        self.mag = np.array([])
        self.layers = []
        self.selected_interval = None  # (layer_idx, interval_idx)
        self._prev_selected = None  # Track previous selection for efficient updates
        self._intervals_dirty = True  # Flag to track if full rebuild needed
        self._interval_data = []  # Cached interval data for selection updates

        # Zoom state
        self.base_x_range = None  # Store original X range for zoom calculations
        self.current_zoom_factor = 1.0  # Current time zoom factor (1.0 = no zoom)
        self.magnitude_zoom_factor = (
            1.0  # Current magnitude zoom factor (1.0 = no zoom)
        )

        # T0 Helper state
        self.t0_helper_active = False
        self.t0_points = []
        self.t0_dragging = None

        # Resize state - always active when interval is selected
        self.resize_dragging = None  # None, 'left', 'right'

        # Resize visuals - borders shown when interval selected
        self.resize_border_lines = []  # List of vertical lines for borders

        # Connect events
        self.canvas.events.mouse_press.connect(self._on_mouse_press)
        self.canvas.events.mouse_release.connect(self._on_mouse_release)
        self.canvas.events.mouse_move.connect(self._on_mouse_move)

    def _canvas_to_data(self, canvas_pos):
        tr = self.canvas.scene.node_transform(self.view.scene)
        pos = tr.map(canvas_pos)
        return pos[0], pos[1]

    def set_title(self, fname: str, algo_type: Algorithm):
        title = ""
        if fname:
            title = f"{os.path.basename(fname)} | {algo_type.value}"
        self.title_text.text = title

    def set_data(self, JD, mag, reset_zoom=True):
        data_changed = not (
            np.array_equal(self.JD, JD) and np.array_equal(self.mag, mag)
        )

        self.JD = JD
        self.mag = mag

        # Reset zoom state only when data actually changes or explicitly requested
        if reset_zoom or data_changed:
            self.base_x_range = None
            self.current_zoom_factor = 1.0
            self.magnitude_zoom_factor = 1.0

        self._update_plot()

    def set_layers(self, layers):
        """Set the layers with intervals."""
        self.layers = layers
        self._intervals_dirty = True
        self._update_intervals()

    def _update_plot(self):
        """Redraw the main scatter plot - just sets camera range.
        Actual scatter is created in _update_intervals for proper coloring."""
        # Clear old scatter (will be recreated in _update_intervals)
        if self.scatter is not None:
            self.scatter.parent = None
            self.scatter = None

        if len(self.JD) == 0:
            return

        # Store base X range if not already set
        if self.base_x_range is None:
            x_margin = (self.JD.max() - self.JD.min()) * 0.02
            self.base_x_range = (self.JD.min() - x_margin, self.JD.max() + x_margin)

        x_range = self._get_zoomed_x_range()
        y_range = self._get_zoomed_y_range()
        self.view.camera.set_range(
            x=x_range,
            y=y_range,
        )

    def _update_intervals(self):
        """Redraw all points and interval visualizations.

        Uses a SINGLE scatter with per-point colors to avoid z-ordering issues:
        - Points NOT in intervals: black
        - Points in intervals: interval color
        """
        if not self._intervals_dirty:
            # Just update selection appearance without rebuilding
            self._update_selection_appearance()
            return

        self._intervals_dirty = False

        # Clear old visuals
        if self.scatter is not None:
            self.scatter.parent = None
            self.scatter = None
        if self.interval_mesh is not None:
            self.interval_mesh.parent = None
            self.interval_mesh = None
        # interval_scatter is no longer used - we use single scatter
        if self.interval_scatter is not None:
            self.interval_scatter.parent = None
            self.interval_scatter = None

        self.interval_metadata = []
        self._interval_data = []

        if len(self.JD) == 0:
            self.canvas.update()
            return

        y_min = -self.mag.max() - (self.mag.max() - self.mag.min()) * 0.1
        y_max = -self.mag.min() + (self.mag.max() - self.mag.min()) * 0.1

        # Collect mesh vertices and colors for interval rectangles (like axvspan in v4.0)
        all_vertices = []
        all_faces = []
        all_mesh_colors = []
        vertex_offset = 0

        # Pre-allocate point colors array - all black initially
        black_color = Color(COLORS["point_data"]).rgba
        point_colors = np.empty((len(self.JD), 4), dtype=np.float32)
        point_colors[:] = black_color

        for lay_idx, layer in enumerate(self.layers):
            starts = layer["intervals"]["start"]
            finishes = layer["intervals"]["finish"]

            for i, (s, f) in enumerate(zip(starts, finishes)):
                if s >= len(self.JD) or f >= len(self.JD):
                    continue

                xs = self.JD[s]
                xf = self.JD[f]

                # Use a larger prime multiplier (7) for better color distribution
                # Add layer offset to ensure different layers use different color subsets
                color_idx = (lay_idx * 7 + i * 5) % len(INTERVAL_COLORS)
                color = INTERVAL_COLORS[color_idx]

                # Check if selected
                is_selected = self.selected_interval == (lay_idx, i)
                alpha = 0.4 if is_selected else 0.15

                # Store interval data for later selection updates
                self._interval_data.append(
                    {
                        "layer_idx": lay_idx,
                        "int_idx": i,
                        "start_idx": s,
                        "finish_idx": f,
                        "color": color,
                        "vertex_start": vertex_offset,
                    }
                )

                # Create full-height rectangle spanning the entire Y range (like axvspan)
                rect_verts = np.array(
                    [
                        [xs, y_min, 0],
                        [xf, y_min, 0],
                        [xf, y_max, 0],
                        [xs, y_max, 0],
                    ],
                    dtype=np.float32,
                )
                all_vertices.append(rect_verts)

                # Two triangles for rectangle
                faces = np.array(
                    [
                        [vertex_offset, vertex_offset + 1, vertex_offset + 2],
                        [vertex_offset, vertex_offset + 2, vertex_offset + 3],
                    ],
                    dtype=np.uint32,
                )
                all_faces.append(faces)

                # Color with alpha for mesh
                c = Color(color)
                c.alpha = alpha
                rgba = c.rgba
                for _ in range(4):
                    all_mesh_colors.append(rgba)

                vertex_offset += 4

                # Set point colors for this interval
                interval_color = Color(color).rgba
                point_colors[s : f + 1] = interval_color

                self.interval_metadata.append((lay_idx, i, xs, xf))

        # Create mesh for all interval rectangles
        if all_vertices:
            vertices = np.vstack(all_vertices)
            faces = np.vstack(all_faces)
            colors = np.array(all_mesh_colors, dtype=np.float32)

            self.interval_mesh = visuals.Mesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=colors,
                parent=self.view.scene,
            )

        # Create SINGLE scatter for ALL points with per-point colors
        pos = np.column_stack([self.JD, -self.mag])
        self.scatter = visuals.Markers(parent=self.view.scene)
        self.scatter.set_data(
            pos,
            face_color=point_colors,
            edge_color=None,  # No edge to avoid white borders on overlapping points
            size=6,
        )
        # Ensure points render on top of interval rectangles
        self.scatter.set_gl_state(depth_test=False)

        self.canvas.update()

    def _update_selection_appearance(self):
        """Update selection highlighting by modifying mesh vertex colors."""
        if self.interval_mesh is None or not self._interval_data:
            self._prev_selected = self.selected_interval
            return

        # Find intervals that need color updates
        to_update = []
        if self._prev_selected is not None:
            to_update.append((self._prev_selected, False))  # Deselect old
        if self.selected_interval is not None:
            to_update.append((self.selected_interval, True))  # Select new

        if not to_update:
            return

        # Get current vertex colors from mesh
        mesh_data = self.interval_mesh.mesh_data
        colors = mesh_data.get_vertex_colors()
        if colors is None:
            self._prev_selected = self.selected_interval
            return

        colors = colors.copy()

        for key, is_selected in to_update:
            # Find the interval data
            for data in self._interval_data:
                if (data["layer_idx"], data["int_idx"]) == key:
                    alpha = 0.4 if is_selected else 0.15
                    c = Color(data["color"])
                    c.alpha = alpha
                    rgba = c.rgba
                    # Update 4 vertices for this rectangle
                    start = data["vertex_start"]
                    for vi in range(4):
                        colors[start + vi] = rgba
                    break

        # Update mesh with new colors
        self.interval_mesh.set_data(
            vertices=mesh_data.get_vertices(),
            faces=mesh_data.get_faces(),
            vertex_colors=colors,
        )

        self._prev_selected = self.selected_interval
        self.canvas.update()

    def _on_mouse_press(self, event):
        """Handle mouse press events."""
        if event.button != 1:  # Left click only
            return

        # Transform to data coordinates
        x_data, y_data = self._canvas_to_data(event.pos)
        y_data = -y_data  # Invert Y back (data is stored as -mag)

        if self.t0_helper_active:
            # Check if clicking near existing T0 marker
            for i, (px, py) in enumerate(self.t0_points):
                if abs(x_data - px) < 0.5 and abs(y_data - py) < 0.02:
                    self.t0_dragging = i
                    return
            self._add_t0_point(x_data, y_data)
            return

        if self.selected_interval is not None:
            # Check if clicking near borders of selected interval
            lay_idx, int_idx = self.selected_interval
            if lay_idx < len(self.layers) and int_idx < len(
                self.layers[lay_idx]["intervals"]["start"]
            ):
                s_idx = self.layers[lay_idx]["intervals"]["start"][int_idx]
                f_idx = self.layers[lay_idx]["intervals"]["finish"][int_idx]
                xs = self.JD[s_idx]
                xf = self.JD[f_idx]

                # Define border tolerance (in data coordinates)
                border_tolerance = (xf - xs) * 0.1  # 10% of interval width
                if border_tolerance < 0.01:  # minimum tolerance
                    border_tolerance = 0.01

                # Check if near left border
                if abs(x_data - xs) < border_tolerance:
                    self.resize_dragging = "left"
                    # Disable camera interaction during border dragging
                    self.view.camera.interactive = False
                    return
                # Check if near right border
                elif abs(x_data - xf) < border_tolerance:
                    self.resize_dragging = "right"
                    # Disable camera interaction during border dragging
                    self.view.camera.interactive = False
                    return

        # Check if clicking on an interval
        for lay_idx, int_idx, xs, xf in self.interval_metadata:
            # Get the Y range for this interval (same as used in _update_intervals)
            y_min = -self.mag.max() - (self.mag.max() - self.mag.min()) * 0.1
            y_max = -self.mag.min() + (self.mag.max() - self.mag.min()) * 0.1

            # Check if click is within both X and Y bounds of the interval rectangle
            if xs <= x_data <= xf and y_min <= -y_data <= y_max:
                old_selection = self.selected_interval
                self.selected_interval = (lay_idx, int_idx)
                if old_selection != self.selected_interval:
                    self._update_selection_appearance()
                self.interval_selected.emit(lay_idx, int_idx)
                return

        # Click on empty space - deselect
        if self.selected_interval is not None:
            self.selected_interval = None
            self._update_selection_appearance()
            self._hide_resize_borders()
        self.point_clicked.emit(x_data, y_data)

    def _on_mouse_release(self, event):
        """Handle mouse release."""
        if self.t0_dragging is not None:
            self.t0_dragging = None
        if self.resize_dragging is not None:
            self.resize_dragging = None
            # Re-enable camera interaction after border dragging
            self.view.camera.interactive = True
            # Rebuild the mesh now that dragging is complete
            if self._intervals_dirty:
                self._update_intervals()

    def _on_mouse_move(self, event):
        """Handle mouse move for T0 helper and resize dragging."""
        # Transform to data coordinates
        x, y = self._canvas_to_data(event.pos)
        y = -y  # Invert Y back (data is stored as -mag)

        if self.t0_dragging is not None:
            idx = self.t0_dragging
            if 0 <= idx < len(self.t0_points):
                self.t0_points[idx] = (x, y)
                self._update_t0_markers()
            return

        if self.resize_dragging is not None:
            self._handle_resize_drag(x)
            return

        # Handle cursor changes for resize mode (always active when interval selected)
        if self.selected_interval is not None:
            self._update_resize_cursor(x)

    def _handle_resize_drag(self, x_data):
        """Handle dragging of interval borders."""
        if self.selected_interval is None or self.resize_dragging is None:
            return

        lay_idx, int_idx = self.selected_interval
        if lay_idx >= len(self.layers) or int_idx >= len(
            self.layers[lay_idx]["intervals"]["start"]
        ):
            return

        # Find the nearest data index to the mouse position
        nearest_idx = np.searchsorted(self.JD, x_data)
        if nearest_idx >= len(self.JD):
            nearest_idx = len(self.JD) - 1
        elif nearest_idx > 0 and abs(self.JD[nearest_idx - 1] - x_data) < abs(
            self.JD[nearest_idx] - x_data
        ):
            nearest_idx -= 1

        # Get current interval bounds
        s_idx = self.layers[lay_idx]["intervals"]["start"][int_idx]
        f_idx = self.layers[lay_idx]["intervals"]["finish"][int_idx]

        # Update the appropriate bound
        if self.resize_dragging == "left":
            # Ensure we don't go beyond the right bound or out of data range
            new_s_idx = max(0, min(nearest_idx, f_idx - 1))
            self.layers[lay_idx]["intervals"]["start"][int_idx] = new_s_idx
        elif self.resize_dragging == "right":
            # Ensure we don't go beyond the left bound or out of data range
            new_f_idx = min(len(self.JD) - 1, max(nearest_idx, s_idx + 1))
            self.layers[lay_idx]["intervals"]["finish"][int_idx] = new_f_idx

        # Mark for update - will rebuild mesh when drag ends
        self._intervals_dirty = True
        self._update_resize_borders_positions()  # Update border lines positions only

    def _update_resize_cursor(self, x_data):
        """Update cursor based on mouse position relative to borders."""
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QCursor

        if self.selected_interval is None:
            self.canvas.native.setCursor(QCursor(Qt.ArrowCursor))
            return

        lay_idx, int_idx = self.selected_interval
        if lay_idx >= len(self.layers) or int_idx >= len(
            self.layers[lay_idx]["intervals"]["start"]
        ):
            self.canvas.native.setCursor(QCursor(Qt.ArrowCursor))
            return

        s_idx = self.layers[lay_idx]["intervals"]["start"][int_idx]
        f_idx = self.layers[lay_idx]["intervals"]["finish"][int_idx]
        xs = self.JD[s_idx]
        xf = self.JD[f_idx]

        # Define border tolerance
        border_tolerance = (xf - xs) * 0.1
        if border_tolerance < 0.01:
            border_tolerance = 0.01

        # Check if near left or right border
        if abs(x_data - xs) < border_tolerance or abs(x_data - xf) < border_tolerance:
            self.canvas.native.setCursor(
                QCursor(Qt.SizeHorCursor)
            )  # Horizontal resize cursor
        else:
            self.canvas.native.setCursor(QCursor(Qt.ArrowCursor))

    def _add_t0_point(self, x, y):
        """Add a T0 helper point."""
        if len(self.t0_points) >= 2:
            self.t0_points.pop(0)
        self.t0_points.append((x, y))
        self._update_t0_markers()

        # Emit T0 update signal if we have two points
        if len(self.t0_points) == 2:
            t0_val = self.get_t0_value()
            if t0_val is not None:
                self.t0_updated.emit(t0_val)

    def _update_t0_markers(self):
        """Update T0 marker visuals."""
        # Remove old markers
        if self.t0_markers is not None:
            self.t0_markers.parent = None
            self.t0_markers = None
        if self.t0_line is not None:
            self.t0_line.parent = None
            self.t0_line = None

        if len(self.t0_points) == 0:
            self.canvas.update()
            return

        # Create markers - red like v4.0
        pos = np.array([[p[0], -p[1]] for p in self.t0_points])
        self.t0_markers = visuals.Markers(parent=self.view.scene)
        self.t0_markers.set_data(
            pos,
            face_color=COLORS["t0_marker"],
            edge_color="white",
            size=15,
            edge_width=2,
        )

        # Draw line between points
        if len(self.t0_points) == 2:
            line_pos = np.array(
                [
                    [self.t0_points[0][0], -self.t0_points[0][1]],
                    [self.t0_points[1][0], -self.t0_points[1][1]],
                ]
            )
            self.t0_line = visuals.Line(
                pos=line_pos, color=COLORS["t0_marker"], width=2, parent=self.view.scene
            )

        self.canvas.update()

    def get_t0_value(self):
        """Calculate T0 from the two helper points."""
        if len(self.t0_points) == 2:
            return abs(self.t0_points[1][0] - self.t0_points[0][0])
        return None

    def clear_t0_helpers(self):
        """Clear T0 helper points."""
        self.t0_points = []
        self._update_t0_markers()

    def _clear_resize_state(self):
        """Clear resize mode state."""
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QCursor

        self.resize_dragging = None
        self.resize_cursor_pos = None
        self._hide_resize_borders()
        self.canvas.native.setCursor(QCursor(Qt.ArrowCursor))  # Reset cursor

    def _show_resize_borders(self):
        """Show or update border lines for the selected interval."""
        if self.selected_interval is None:
            self._hide_resize_borders()
            return

        lay_idx, int_idx = self.selected_interval
        if lay_idx >= len(self.layers) or int_idx >= len(
            self.layers[lay_idx]["intervals"]["start"]
        ):
            self._hide_resize_borders()
            return

        s_idx = self.layers[lay_idx]["intervals"]["start"][int_idx]
        f_idx = self.layers[lay_idx]["intervals"]["finish"][int_idx]
        xs = self.JD[s_idx]
        xf = self.JD[f_idx]

        # Get Y range for the lines
        y_min = -self.mag.max() - (self.mag.max() - self.mag.min()) * 0.1
        y_max = -self.mag.min() + (self.mag.max() - self.mag.min()) * 0.1

        # Always hide existing borders first to prevent duplicates
        self._hide_resize_borders()

        # Create exactly 2 border lines - make them thicker and more visible
        left_line = visuals.Line(
            pos=np.array([[xs, y_min], [xs, y_max]]),
            color=COLORS["accent"],
            width=2,
            parent=self.view.scene,
        )
        left_line.set_gl_state(depth_test=False)
        self.resize_border_lines.append(left_line)

        right_line = visuals.Line(
            pos=np.array([[xf, y_min], [xf, y_max]]),
            color=COLORS["accent"],
            width=2,
            parent=self.view.scene,
        )
        right_line.set_gl_state(depth_test=False)
        self.resize_border_lines.append(right_line)

        self.canvas.update()

    def _update_resize_borders_positions(self):
        """Update positions of existing border lines without recreating them."""
        if self.selected_interval is None:
            return

        lay_idx, int_idx = self.selected_interval
        if lay_idx >= len(self.layers) or int_idx >= len(
            self.layers[lay_idx]["intervals"]["start"]
        ):
            return

        if len(self.resize_border_lines) < 2:
            return  # Borders not created yet

        s_idx = self.layers[lay_idx]["intervals"]["start"][int_idx]
        f_idx = self.layers[lay_idx]["intervals"]["finish"][int_idx]
        xs = self.JD[s_idx]
        xf = self.JD[f_idx]

        # Get Y range for the lines
        y_min = -self.mag.max() - (self.mag.max() - self.mag.min()) * 0.1
        y_max = -self.mag.min() + (self.mag.max() - self.mag.min()) * 0.1

        # Update existing lines
        self.resize_border_lines[0].set_data(pos=np.array([[xs, y_min], [xs, y_max]]))
        self.resize_border_lines[1].set_data(pos=np.array([[xf, y_min], [xf, y_max]]))

    def _hide_resize_borders(self):
        """Hide border lines."""
        for line in self.resize_border_lines:
            if line.parent is not None:
                line.parent = None
        self.resize_border_lines = []
        self.canvas.update()

    def _get_zoomed_x_range(self):
        """Get the current zoomed X range based on zoom factor."""
        if self.base_x_range is None or self.current_zoom_factor == 1.0:
            return self.base_x_range

        # Calculate center of current range
        x_min, x_max = self.base_x_range
        center = (x_min + x_max) / 2

        # Calculate new range based on zoom factor
        # Zoom factor > 1.0 means zoom in (smaller range)
        # Zoom factor < 1.0 means zoom out (larger range)
        range_width = (x_max - x_min) / self.current_zoom_factor
        new_x_min = center - range_width / 2
        new_x_max = center + range_width / 2

        return (new_x_min, new_x_max)

    def _get_zoomed_y_range(self):
        """Get the current zoomed Y range based on magnitude zoom factor."""
        if len(self.mag) == 0:
            return (-1, 1)

        mag_min, mag_max = self.mag.min(), self.mag.max()

        if self.magnitude_zoom_factor == 1.0:
            # Add margin for full range
            y_margin = (mag_max - mag_min) * 0.1
            return (-mag_max - y_margin, -mag_min + y_margin)

        # Calculate center of magnitude range (note: Y axis is inverted)
        center = -(mag_min + mag_max) / 2

        # Calculate new range based on zoom factor
        # Zoom factor > 1.0 means zoom in (smaller range)
        # Zoom factor < 1.0 means zoom out (larger range)
        range_height = (mag_max - mag_min) / self.magnitude_zoom_factor
        y_margin = range_height * 0.1  # Add some margin

        new_y_min = center - range_height / 2 - y_margin
        new_y_max = center + range_height / 2 + y_margin

        return (new_y_min, new_y_max)

    def set_time_zoom(self, zoom_factor: float):
        """Set the zoom factor for the time axis."""
        if zoom_factor <= 0:
            return

        self.current_zoom_factor = zoom_factor
        self._update_camera_range()

    def set_magnitude_zoom(self, zoom_factor: float):
        """Set the zoom factor for the magnitude axis."""
        if zoom_factor <= 0:
            return

        self.magnitude_zoom_factor = zoom_factor
        self._update_camera_range()

    def _update_camera_range(self):
        """Update the camera range based on current zoom factors."""
        if len(self.JD) == 0:
            return

        x_range = self._get_zoomed_x_range()
        y_range = self._get_zoomed_y_range()

        self.view.camera.set_range(x=x_range, y=y_range)

    def reset_time_zoom(self):
        """Reset time axis zoom to show full range."""
        self.current_zoom_factor = 1.0
        self._update_camera_range()

    def reset_magnitude_zoom(self):
        """Reset magnitude axis zoom to show full range."""
        self.magnitude_zoom_factor = 1.0
        self._update_camera_range()

    def reset_view(self):
        """Reset camera to show all data."""
        if len(self.JD) == 0:
            return

        # Reset zoom factors
        self.current_zoom_factor = 1.0
        self.magnitude_zoom_factor = 1.0

        x_margin = (self.JD.max() - self.JD.min()) * 0.02
        self.base_x_range = (self.JD.min() - x_margin, self.JD.max() + x_margin)

        self._update_camera_range()

    def clear_selection(self):
        """Clear interval selection."""
        if self.selected_interval is not None:
            self.selected_interval = None
            self._update_selection_appearance()
            self._hide_resize_borders()
