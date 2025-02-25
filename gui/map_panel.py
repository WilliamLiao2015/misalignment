import cv2
import numpy as np

from trajdata.maps import VectorMap
from PIL import Image
from PySide6.QtWidgets import QStackedLayout, QWidget
from PySide6.QtGui import QImage, QPixmap

from utils.map import flatten_map
from .map_viewer import MapViewer

class MapPanel(QWidget):
    def __init__(self, window, mode="vector"):
        super().__init__()
        self.setMouseTracking(True)

        self.window = window
        self.window.commands["update_map"] = self.update_map
        self.mode = mode

        self.vector_map = None
        self.image = None

        self.selected_lanes = []

        layout = QStackedLayout()

        self.viewer = MapViewer(self, self.mode)

        layout.addWidget(self.viewer)

        self.setLayout(layout)

    def mousePressEvent(self, event):
        scene_pos = self.viewer.mapToScene(event.pos())
        lanes = self.vector_map.get_lanes_within(np.asarray([scene_pos.x(), scene_pos.y(), 0]), 1.5)

        continue_lanes = False

        for lane in lanes:
            if len(self.selected_lanes) > 0 and lane.id in self.selected_lanes[-1].next_lanes:
                self.selected_lanes.append(lane)
                continue_lanes = True
                break

        if not continue_lanes and len(lanes) > 0:
            self.selected_lanes = [lanes[0]]

        self.viewer.clearInteractiveOverlay()
        self.viewer.drawSelectedLanes(self.selected_lanes, draw_reachable=True)

    def update_map(self, map_name):
        """Update the map display."""
        print(f"{self.window.env_name}:{map_name}")

        self.vector_map: VectorMap = self.window.map_api.get_map(f"{self.window.env_name}:{map_name}")
        self.vector_map = flatten_map(self.vector_map)
        self.window.vector_map = self.vector_map

        if self.mode == "raster":
            self.image = self.vector_map.rasterize(resolution=10, incl_lane_area=False)
            alpha = (~np.all(self.image == [0, 0, 0], axis=-1) * 255).astype(np.uint8)
            self.image = np.dstack((self.image * 255, alpha)).astype(np.uint8)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2RGBA)
            self.image = Image.fromarray(self.image)

            image = QImage(self.image.tobytes(), self.image.size[0], self.image.size[1], QImage.Format.Format_RGBA8888)
            pixmap = QPixmap.fromImage(image)

            self.viewer.setMap(pixmap)
        elif self.mode == "vector":
            self.viewer.drawMap(self.vector_map)
