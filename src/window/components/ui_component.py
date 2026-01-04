from abc import ABC, abstractmethod
from typing import Optional, Any
from PySide6.QtWidgets import QWidget


class UIComponent(ABC):
    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        self.title = title
        self.parent = parent
        self.widget: Optional[QWidget] = None
        self._signals_connected = False

    @abstractmethod
    def create_widget(self) -> QWidget:
        pass

    def get_widget(self) -> QWidget:
        if self.widget is None:
            if self.title:
                # Wrap in a group box
                from PySide6.QtWidgets import QGroupBox, QVBoxLayout

                group_box = QGroupBox(self.title)
                layout = QVBoxLayout(group_box)
                layout.addWidget(self.create_widget())
                self.widget = group_box
            else:
                # No group box wrapper
                self.widget = self.create_widget()
        return self.widget

    def connect_signals(self, window: Any) -> None:
        self._signals_connected = True

    def update_state(self, **kwargs) -> None:
        pass

    def set_enabled(self, enabled: bool) -> None:
        if self.widget:
            self.widget.setEnabled(enabled)
