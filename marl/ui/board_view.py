import math
import os

from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPolygonItem, QGraphicsEllipseItem,
    QGraphicsLineItem, QGraphicsTextItem, QWidget, QApplication, QVBoxLayout
)
from PyQt6.QtGui import QBrush, QPen, QColor, QPolygonF, QFont, QPainter, QPixmap
from PyQt6.QtCore import QPointF, Qt

from marl.model.CatanGame import CatanGame


class NodeItem(QGraphicsEllipseItem):
    RADIUS = 8.0

    def __init__(self, pos: QPointF, index: int = -1):
        r = NodeItem.RADIUS
        super().__init__(pos.x() - r, pos.y() - r, r * 2, r * 2)
        self.setBrush(QBrush(QColor(255, 255, 255)))
        self.setPen(QPen(Qt.GlobalColor.black, 1))
        self.setZValue(2)
        self.index = index
        self.selected = False

        self.text_item = QGraphicsTextItem(str(index), self)
        self.text_item.setFont(QFont("Arial", 7))
        self.text_item.setDefaultTextColor(Qt.GlobalColor.black)
        self.text_item.setPos(-r / 2, -r / 2)

        self.setAcceptHoverEvents(True)

    def mousePressEvent(self, event):
        view = self.scene().views()[0]

        # If board is waiting for a node → deliver it and exit selection mode
        if view.awaiting_node_callback:
            callback = view.awaiting_node_callback
            view.awaiting_node_callback = None
            callback(self.index)
            event.accept()
            return

        if getattr(view, "selected_item", None) is self:
            self.set_selected(False)
            view.selected_item = None
        else:
            if getattr(view, "selected_item", None):
                view.selected_item.set_selected(False)
            view.selected_item = self
            self.set_selected(True)
        event.accept()

    def hoverEnterEvent(self, event):
        if not self.selected:
            self.setPen(QPen(QColor(200, 120, 0), 2))

    def hoverLeaveEvent(self, event):
        self.update_style()

    def set_selected(self, selected: bool):
        self.selected = selected
        self.update_style()

    def update_style(self):
        if self.selected:
            self.setBrush(QBrush(QColor(255, 200, 0)))
            self.setPen(QPen(QColor(200, 120, 0), 2))
        else:
            self.setBrush(QBrush(QColor(255, 255, 255)))
            self.setPen(QPen(Qt.GlobalColor.black, 1))


class EdgeItem(QGraphicsLineItem):
    WIDTH = 6.0

    def __init__(self, p1: QPointF, p2: QPointF, index: int = -1):
        super().__init__(p1.x(), p1.y(), p2.x(), p2.y())
        self.selected = False
        self.index = index
        self.setZValue(1)
        self.setAcceptHoverEvents(True)
        self.setPen(QPen(QColor(120, 120, 120), EdgeItem.WIDTH, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))

    def mousePressEvent(self, event):
        view = self.scene().views()[0]

        # If board is waiting for an edge → deliver index
        if view.awaiting_edge_callback:
            callback = view.awaiting_edge_callback
            view.awaiting_edge_callback = None
            callback(self.index)
            event.accept()
            return

        if getattr(view, "selected_item", None) is self:
            self.set_selected(False)
            view.selected_item = None
        else:
            if getattr(view, "selected_item", None):
                view.selected_item.set_selected(False)
            view.selected_item = self
            self.set_selected(True)
        event.accept()

    def hoverEnterEvent(self, event):
        if not self.selected:
            pen = self.pen()
            self.setPen(QPen(pen.color(), pen.width() + 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))

    def hoverLeaveEvent(self, event):
        self.update_style()

    def set_selected(self, selected: bool):
        self.selected = selected
        self.update_style()

    def update_style(self):
        if self.selected:
            self.setPen(QPen(QColor(30, 144, 255), EdgeItem.WIDTH + 2,
                             Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        else:
            self.setPen(QPen(QColor(120, 120, 120), EdgeItem.WIDTH,
                             Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))


class HexItem(QGraphicsPolygonItem):
    def __init__(self,
                 center: QPointF,
                 radius: float,
                 fill: QColor = QColor(224, 179, 101),
                 texture_path: str | None = None):
        super().__init__()
        self.center = center
        self.radius = radius

        polygon = self._create_hex_polygon(center, radius)
        self.setPolygon(QPolygonF(polygon))
        self.setPen(QPen(Qt.GlobalColor.black, 1))
        self.setZValue(0)

        # Default flat color
        brush = QBrush(fill)

        if texture_path:
            texture_path = os.path.abspath(os.path.join(os.path.dirname(__file__), texture_path))
            pixmap = QPixmap(texture_path)
            print(f"Loading {texture_path} Exists={os.path.exists(texture_path)} Null={pixmap.isNull()}")

            if not pixmap.isNull():
                size = int(radius * 2.2)
                pixmap = pixmap.scaled(size, size,
                                       Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                       Qt.TransformationMode.SmoothTransformation)
                brush = QBrush(pixmap)
            else:
                raise ValueError(f"⚠️ Failed to load texture: {texture_path}")

            self.setBrush(brush)

    @staticmethod
    def _create_hex_polygon(center: QPointF, radius: float):
        pts = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.radians(angle_deg)
            x = center.x() + radius * math.cos(angle_rad)
            y = center.y() + radius * math.sin(angle_rad)
            pts.append(QPointF(x, y))
        return pts


class BoardView(QGraphicsView):
    ROW_COUNTS = [3, 4, 5, 4, 3]

    def __init__(self, game: CatanGame | None = None, hex_radius: float = 70):
        super().__init__()
        self.game = game
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.Antialiasing)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.hex_radius = hex_radius
        self.selected_item = None
        self.awaiting_node_callback = None
        self.awaiting_edge_callback = None
        self.nodes = []
        self.edges = []

        self._build_board()
        self.setBackgroundBrush(QBrush(QColor(230, 230, 230)))
        self.setMinimumSize(600, 600)

    def _build_board(self):
        r = self.hex_radius
        hex_w = 2 * r
        hex_h = math.sqrt(3) * r
        h_spacing = r * 1.75
        v_spacing = hex_h * 0.88
        max_tiles = max(self.ROW_COUNTS)
        board_width = (max_tiles - 1) * h_spacing + hex_w
        current_y = 0.0
        hex_centers = []

        # calculate hex centers
        for _, tiles in enumerate(self.ROW_COUNTS):
            row_width = (tiles - 1) * h_spacing + hex_w
            start_x = (board_width - row_width) / 2.0 + r
            for i in range(tiles):
                cx = start_x + i * h_spacing
                cy = current_y + hex_h / 2.0
                hex_centers.append(QPointF(cx, cy))
            current_y += v_spacing

        # shift to positive coordinates
        min_x = min(pt.x() for pt in hex_centers)
        min_y = min(pt.y() for pt in hex_centers)
        margin = 40
        translate_x = -min_x + margin
        translate_y = -min_y + margin

        # create hexes, nodes, edges
        node_map = {}
        node_index_counter = 0
        for i, center in enumerate(hex_centers):
            center = QPointF(center.x() + translate_x, center.y() + translate_y)
            hex_item = HexItem(center, r, texture_path=f'./assets/{self.game.board.tiles[i][0]}.jpg')
            self.scene.addItem(hex_item)
            corners = HexItem._create_hex_polygon(center, r)
            # add nodes with deduplication
            node_items = []
            for c in corners:
                node, new_node = self.find_or_create_node(node_map, c, node_index_counter)
                if new_node:
                    node_index_counter += 1
                node_items.append(node)
            for i in range(6):
                line = EdgeItem(corners[i], corners[(i + 1) % 6])
                self.scene.addItem(line)
                self.edges.append(line)

        # number nodes top-left → bottom-right
        sorted_nodes = sorted(node_map.values(), key=lambda n: (n.scenePos().y(), n.scenePos().x()))
        for idx, node in enumerate(sorted_nodes):
            node.index = idx
            node.text_item.setPlainText(str(idx))

        self.nodes = sorted_nodes
        self.scene.setSceneRect(self.scene.itemsBoundingRect())

    def find_or_create_node(self, node_map, c, index_counter):
        EPS = 2.0
        for key, node in node_map.items():
            if (abs(key[0] - c.x()) < EPS and abs(key[1] - c.y()) < EPS):
                return node, False
        # no match → create
        node = NodeItem(c, index_counter)
        node_map[(c.x(), c.y())] = node
        self.scene.addItem(node)
        return node, True

    def expect_node_selection(self, callback):
        """BoardView will call callback(node_index) after user clicks a node."""
        self.awaiting_node_callback = callback
        self.awaiting_edge_callback = None
        if self.selected_item:
            self.selected_item.set_selected(False)
            self.selected_item = None

    def expect_edge_selection(self, callback):
        """BoardView will call callback(edge_index) after user clicks an edge."""
        self.awaiting_edge_callback = callback
        self.awaiting_node_callback = None
        if self.selected_item:
            self.selected_item.set_selected(False)
            self.selected_item = None



# Demo run
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = QWidget()
    layout = QVBoxLayout()
    board = BoardView(hex_radius=60)
    layout.addWidget(board)
    w.setLayout(layout)
    w.setWindowTitle("Catan Board Demo")
    w.resize(900, 800)
    w.show()
    sys.exit(app.exec())
