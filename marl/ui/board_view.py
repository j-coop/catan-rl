import math
import os

from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPolygonItem,
    QGraphicsLineItem, QWidget, QApplication, QVBoxLayout, QGraphicsItem
)
from PyQt6.QtGui import (
    QBrush, QPen, QColor, QPolygonF, 
    QPixmap, QPainter, QPainterPath
)
from PyQt6.QtCore import QPointF, Qt, QRectF

from marl.model.CatanGame import CatanGame
from params.catan_constants import N_EDGES, N_NODES
from params.tiles2edges_adjacency_map import TILES_TO_EDGES
from params.tiles2nodes_adjacency_map import TILES_TO_NODES
from params.edges_list import EDGES_LIST


class NodeItem(QGraphicsItem):
    RADIUS = 8.0

    def __init__(self,
                 pos: QPointF,
                 index: int,
                 owner_color: str | None = None):
        super().__init__()
        self.setPos(pos)
        self.setZValue(2)

        self.index = index
        self.selected = False
        self.hovered = False
        self.owner_color = owner_color

        self.setAcceptHoverEvents(True)

    def boundingRect(self) -> QRectF:
        r = self.RADIUS
        return QRectF(-r, -r, 2 * r, 2 * r)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addEllipse(self.boundingRect())
        return path

    def paint(self, painter: QPainter, option, widget=None):
        if self.owner_color is not None:
            self._draw_settlement(painter)
        else:
            self._draw_empty_node(painter)

    def _draw_empty_node(self, painter: QPainter):
        if self.hovered:
            brush = QBrush(QColor(255, 200, 0))
            pen = QPen(QColor(200, 120, 0), 2)
        else:
            brush = QBrush(QColor(255, 255, 255))
            pen = QPen(Qt.GlobalColor.black, 1)

        painter.setBrush(brush)
        painter.setPen(pen)
        painter.drawEllipse(self.boundingRect())

    def _draw_settlement(self, painter: QPainter):
        fill = QColor(self.owner_color)
        if self.hovered and not self.selected:
            fill = fill.darker(130)
        outline = QColor(0, 0, 0)

        painter.setBrush(QBrush(fill))
        painter.setPen(QPen(outline, 1.5))

        r = self.RADIUS
        scale = 3.0

        path = QPainterPath()
        path.moveTo(-r * 0.5 * scale,  r * 0.4 * scale)
        path.lineTo(-r * 0.5 * scale, -r * 0.1 * scale)
        path.lineTo(0, -r * 0.6 * scale)
        path.lineTo( r * 0.5 * scale, -r * 0.1 * scale)
        path.lineTo( r * 0.5 * scale,  r * 0.4 * scale)
        path.closeSubpath()
        painter.drawPath(path)

    def mousePressEvent(self, event):
        view = self.scene().views()[0]

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
        self.hovered = True
        self.update()

    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.update()

    def set_selected(self, selected: bool):
        self.selected = selected
        self.update()


class EdgeItem(QGraphicsLineItem):
    WIDTH = 8.0
    DEFAULT_COLOR = QColor(120, 120, 120)
    SELECTED_COLOR = QColor(30, 144, 255)
    HOVER_DARKEN_FACTOR = 130

    def __init__(self,
                 p1: QPointF, p2: QPointF,
                 owner_color: str | None = None,
                 index: int = -1):
        super().__init__(p1.x(), p1.y(), p2.x(), p2.y())

        self.selected = False
        self.hovered = False
        self.owner_color = QColor(owner_color) if owner_color else None
        self.index = index

        self.setZValue(1)
        self.setAcceptHoverEvents(True)
        self.update_style()

    def base_color(self) -> QColor:
        return self.owner_color or EdgeItem.DEFAULT_COLOR

    def hoverEnterEvent(self, event):
        self.hovered = True
        self.update_style()

    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.update_style()

    def mousePressEvent(self, event):
        view = self.scene().views()[0]

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

    def set_selected(self, selected: bool):
        self.selected = selected
        self.update_style()

    def update_style(self):
        if self.selected:
            color = EdgeItem.SELECTED_COLOR
            width = EdgeItem.WIDTH + 2
        else:
            color = self.base_color()
            width = EdgeItem.WIDTH
            if self.hovered:
                color = color.darker(EdgeItem.HOVER_DARKEN_FACTOR)

        self.setPen(QPen(
            color,
            width,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap
        ))


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
        node_creation_map = [False] * N_NODES
        edge_creation_map = [False] * N_EDGES
        hex_centers = sorted(
            hex_centers,
            key=lambda p: (round(p.y(), 6), round(p.x(), 6))
        )
        for i, center in enumerate(hex_centers):
            center = QPointF(center.x() + translate_x, center.y() + translate_y)
            hex_item = HexItem(center, r, texture_path=f'./assets/{self.game.board.tiles[i][0]}.jpg')
            self.scene.addItem(hex_item)
            corners = HexItem._create_hex_polygon(center, r)
            sorted_corners = self._sort_corners(corners)

            indices = TILES_TO_NODES[i]
            for j in range(6):
                corner = sorted_corners[j]
                index = indices[j]
                if not node_creation_map[index]:
                    node = self.create_node(corner, index)
                    self.nodes.append(node)
                    self.scene.addItem(node)
                    node_creation_map[index] = True

            hex_edges = TILES_TO_EDGES[i]
            for j in range(6):
                edge_idx = EDGES_LIST.index(hex_edges[j])
                if not edge_creation_map[edge_idx]:
                    edge_creation_map[edge_idx] = True
                    edge = self.create_edge(
                        sorted_corners[j],
                        sorted_corners[(j + 1) % 6],
                        edge_idx
                    )
                    self.scene.addItem(edge)
                    self.edges.append(edge)

        self.scene.setSceneRect(self.scene.itemsBoundingRect())

    def create_node(self, corner, index):
        owner = self.game.board.nodes[index]
        if owner is None:
            node = NodeItem(corner, index)
        else:
            color = self.game.get_player(owner).color
            node = NodeItem(corner, index, owner_color=color)
        self.scene.addItem(node)
        return node
    
    def create_edge(self, p1: QPointF, p2: QPointF, index: int):
        owner = self.game.board.edges[index]
        if owner is None:
            edge = EdgeItem(p1, p2, index=index)
        else:
            color = self.game.get_player(owner).color
            edge = EdgeItem(p1, p2, owner_color=color, index=index)

        self.scene.addItem(edge)
        return edge

    def find_or_create_edge(self, node_map, c, index_counter):
        EPS = 2.0
        for key, node in node_map.items():
            if (abs(key[0] - c.x()) < EPS and abs(key[1] - c.y()) < EPS):
                return node, False
        # no match → create
        node = NodeItem(c, index_counter)
        node_map[(c.x(), c.y())] = node
        self.scene.addItem(node)
        return node, True

    def _sort_corners(self, corners):
        s = sorted(
            corners,
            key=lambda p: (round(p.y(), 2), round(p.x(), 2))
        )
        s[0], s[1] = s[1], s[0]
        s[3], s[4] = s[4], s[3]
        s[4], s[5] = s[5], s[4]
        return s

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
