from PySide6.QtCore import Qt, QPoint, QRectF, Signal
from PySide6.QtGui import QPixmap, QColor, QCursor, QFont, QPalette, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsView, QFrame, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPathItem, QGraphicsItemGroup, QGraphicsTextItem

SCALE_FACTOR = 1.05

class MapViewer(QGraphicsView):
    coordinatesChanged = Signal(QPoint)

    def __init__(self, parent, mode="vector"):
        super().__init__(parent)

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        self.mode = mode
        self.interactive_overlay = None
        self.vector_map = None

        self._zoom = 0
        self._pinned = False
        self._empty = True
        self._scene = QGraphicsScene(self)

        if self.mode == "raster":
            self._photo = QGraphicsPixmapItem()
            self._photo.setShapeMode(QGraphicsPixmapItem.ShapeMode.BoundingRectShape)
            self._scene.addItem(self._photo)

        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def hasMap(self):
        return not self._empty

    def resetView(self, scale=1):
        rect = QRectF(self._photo.pixmap().rect() if self.mode == "raster" else self._scene.itemsBoundingRect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if (scale := max(1, scale)) == 1:
                self._zoom = 0
            if self.mode == "raster" and self.hasMap() or self.mode == "vector":
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(), viewrect.height() / scenerect.height()) * scale
                self.scale(factor, factor)
                if not self.zoomPinned():
                    self.centerOn(self._photo if self.mode == "raster" else self._scene.items()[0])
                self.updateCoordinates()

    def setMap(self, pixmap=None):
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._photo.setPixmap(QPixmap())
        if not (self.zoomPinned() and self.hasMap()):
            self._zoom = 0
        self.resetView(SCALE_FACTOR ** self._zoom)

    def zoomLevel(self):
        return self._zoom

    def zoomPinned(self):
        return self._pinned

    def setZoomPinned(self, enable):
        self._pinned = bool(enable)

    def zoom(self, step):
        zoom = max(0, self._zoom + (step := int(step)))
        if zoom != self._zoom:
            self._zoom = zoom
            if self._zoom > 0:
                if step > 0:
                    factor = SCALE_FACTOR ** step
                else:
                    factor = 1 / SCALE_FACTOR ** abs(step)
                self.scale(factor, factor)
            else:
                self.resetView()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.zoom(delta and delta // abs(delta) * (1 + abs(delta) // 120))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resetView()

    def toggleDragMode(self):
        if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        elif self.mode == "vector" or not self._photo.pixmap().isNull():
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def updateCoordinates(self, pos=None):
        if self.mode == "vector" or self._photo.isUnderMouse():
            if pos is None:
                pos = self.mapFromGlobal(QCursor.pos())
            point = self.mapToScene(pos).toPoint()
        else:
            point = QPoint()
        self.coordinatesChanged.emit(point)

    def mouseMoveEvent(self, event):
        self.updateCoordinates(event.position().toPoint())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.coordinatesChanged.emit(QPoint())
        super().leaveEvent(event)

    def drawMap(self, vector_map, draw_text=False):
        self._scene.clear()

        for lane in vector_map.lanes:
            center_points = lane.center.xy
            left_edge_points = lane.left_edge.xy if lane.left_edge is not None else None
            right_edge_points = lane.right_edge.xy if lane.right_edge is not None else None

            path = QPainterPath()
            path.moveTo(center_points[0][0], center_points[0][1])
            for point in center_points[1:]:
                path.lineTo(point[0], point[1])

            path_item = QGraphicsPathItem(path)
            path_item.setPen(QPen(QColor(0, 0, 0, 64), 0.3, c=Qt.PenCapStyle.RoundCap, j=Qt.PenJoinStyle.RoundJoin))
            self._scene.addItem(path_item)

            if left_edge_points is not None:
                left_path = QPainterPath()
                left_path.moveTo(left_edge_points[0][0], left_edge_points[0][1])
                for point in left_edge_points[1:]:
                    left_path.lineTo(point[0], point[1])

                left_path_item = QGraphicsPathItem(left_path)
                left_path_item.setPen(QPen(QColor(0, 0, 0, 255), 0.5, c=Qt.PenCapStyle.RoundCap, j=Qt.PenJoinStyle.RoundJoin))
                self._scene.addItem(left_path_item)

            if right_edge_points is not None:
                right_path = QPainterPath()
                right_path.moveTo(right_edge_points[0][0], right_edge_points[0][1])
                for point in right_edge_points[1:]:
                    right_path.lineTo(point[0], point[1])

                right_path_item = QGraphicsPathItem(right_path)
                right_path_item.setPen(QPen(QColor(0, 0, 0, 255), 0.5, c=Qt.PenCapStyle.RoundCap, j=Qt.PenJoinStyle.RoundJoin))
                self._scene.addItem(right_path_item)

            if draw_text:
                text_item = QGraphicsTextItem(lane.id)
                text_item.setPos(center_points[0][0], center_points[0][1])
                text_item.setDefaultTextColor(QColor(0, 0, 0, 64))
                text_item.setFont(QFont("Consolas", 2))
                self._scene.addItem(text_item)

        self.vector_map = vector_map

        self._empty = False
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.resetView(SCALE_FACTOR ** self._zoom)

    def clearInteractiveOverlay(self):
        try: self._scene.removeItem(self.interactive_overlay)
        except: pass

        self.interactive_overlay = QGraphicsItemGroup()
        self._scene.addItem(self.interactive_overlay)

    def drawSelectedLanes(self, lanes, draw_reachable=False, color=(255, 0, 0, 255)):
        for lane in lanes:
            center_points = lane.center.xy

            path = QPainterPath()
            path.moveTo(center_points[0][0], center_points[0][1])
            for point in center_points[1:]:
                path.lineTo(point[0], point[1])

            path_item = QGraphicsPathItem(path)
            path_item.setPen(QPen(QColor(*color), 0.5, c=Qt.PenCapStyle.RoundCap, j=Qt.PenJoinStyle.RoundJoin))
            self.interactive_overlay.addToGroup(path_item)

            text_item = QGraphicsTextItem(lane.id)
            text_item.setPos(center_points[0][0], center_points[0][1])
            text_item.setDefaultTextColor(QColor(*color))
            text_item.setFont(QFont("Consolas", 2))
            self.interactive_overlay.addToGroup(text_item)

        if draw_reachable:
            self.drawSelectedLanes([self.vector_map.get_road_lane(lane_id) for lane_id in lane.next_lanes], draw_reachable=False, color=(255, 255, 0, 255))

        # self._empty = False
        # self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # self.resetView(SCALE_FACTOR ** self._zoom)
