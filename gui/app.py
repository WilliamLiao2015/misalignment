import os
import re
import sys

from pathlib import Path

from trajdata.maps import MapAPI
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget

from .control_panel import ControlPanel
from .map_panel import MapPanel

cache_path = "~/.unified_data_cache"
# cache_path = "../data"

class MainWindow(QMainWindow):
    def __init__(self, cache_path="~/.unified_data_cache", env_name="waymo_val", methods=["lctgen"]):
        super().__init__()
        self.setWindowTitle("Trajectory Customization")
        self.setFixedSize(QSize(800, 600))

        # Environment Configuration
        self.env_name = env_name
        self.methods = methods

        self.method_name = methods[0]
        self.vector_map = None
        self.query = None
        self.commands = {}

        # Map API Initialization
        self.map_api = MapAPI(Path(cache_path).expanduser())
        self.maps = sorted([
            map_file.replace(".pb", "")
            for map_file in os.listdir(self.map_api.unified_cache_path / env_name / "maps")
            if map_file.endswith(".pb")
        ], key=lambda x: int(re.match(fr"{env_name}_(\d+)", x).group(1)))

        central_widget = QWidget()
        layout = QHBoxLayout(central_widget)

        self.map_panel = MapPanel(self)
        self.control_panel = ControlPanel(self)

        layout.addWidget(self.map_panel)
        layout.addWidget(self.control_panel)

        self.setCentralWidget(central_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow(cache_path=cache_path)
    window.show()

    app.exec()
