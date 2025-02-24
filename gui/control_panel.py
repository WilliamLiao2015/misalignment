from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget, QLabel, QSlider, QComboBox

class ControlPanel(QWidget):
    def __init__(self, window):
        super().__init__()
        self.setMaximumWidth(200)

        self.window = window

        layout = QVBoxLayout()

        self.map_selector = QComboBox()
        self.map_selector.addItems(self.window.maps)
        self.map_selector.setCurrentIndex(-1)
        self.map_selector.currentTextChanged.connect(self.window.commands["update_map"])

        self.slider = QSlider()
        self.slider.setOrientation(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.slider_changed)

        self.label = QLabel("Value: 50")

        layout.addWidget(self.map_selector)
        layout.addWidget(self.slider)
        layout.addWidget(self.label)

        self.setLayout(layout)

    def slider_changed(self):
        self.label.setText(f"Value: {self.slider.value()}")
