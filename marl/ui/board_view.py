# ui/board_view.py
import math
from typing import Dict, Tuple
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPolygonItem, QGraphicsEllipseItem,
    QGraphicsLineItem, QApplication, QWidget, QVBoxLayout
)
from PyQt6.QtGui import QBrush, QPen, QPolygonF, QColor, QPainter
from PyQt6.QtCore import QPointF, Qt


class HexItem(QGraphicsPolygonItem):
    def __init__(self, center: QPointF, radius: float, fill: QColor = QColor(224, 179, 101)):
        super().__init__()
        self.center = center
        self.radius = radius
        self.fill = fill
        self.setPolygon(self._create_hex_polygon(center, radius))
        self.setBrush(QBrush(self.fill))
        self.setPen(QPen(Qt.GlobalColor.black, 1))
        self.setZValue(0)

    @staticmethod
    def _create_hex_polygon(center: QPointF, r: float) -> QPolygonF:
        # pointy-top hexagon (flat sides not horizontal)
        pts = []
        for i in range(6):
            angle_deg = 60 * i - 30  # shift so top has a point
            angle_rad = math.radians(angle_deg)
            x = center.x() + r * math.cos(angle_rad)
            y = center.y() + r * math.sin(angle_rad)
            pts.append(QPointF(x, y))
        return QPolygonF(pts)


class NodeItem(QGraphicsEllipseItem):
    RADIUS = 8.0

    def __init__(self, pos: QPointF, node_id: int):
        r = NodeItem.RADIUS
        super().__init__(pos.x() - r, pos.y() - r, r * 2, r * 2)
        self.setBrush(QBrush(QColor(255, 255, 255)))
        self.setPen(QPen(Qt.GlobalColor.black, 1))
        self.setZValue(2)
        self.node_id = node_id
        self.selected = False
        # Make it hoverable and selectable style-wise
        self.setAcceptHoverEvents(True)

    def mousePressEvent(self, event):
        # toggle selection
        self.selected = not self.selected
        self.update_style()
        print(f"Node clicked: id={self.node_id}, selected={self.selected}")
        # stop further propagation
        event.accept()

    def update_style(self):
        if self.selected:
            self.setBrush(QBrush(QColor(255, 200, 0)))  # highlight color
            self.setPen(QPen(QColor(200, 120, 0), 2))
        else:
            self.setBrush(QBrush(QColor(255, 255, 255)))
            self.setPen(QPen(Qt.GlobalColor.black, 1))

    def hoverEnterEvent(self, event):
        self.setScale(1.1)

    def hoverLeaveEvent(self, event):
        self.setScale(1.0)


class EdgeItem(QGraphicsLineItem):
    WIDTH = 6.0  # visual thickness (we will draw line but use pen width)
    def __init__(self, p1: QPointF, p2: QPointF, edge_id: int):
        super().__init__(p1.x(), p1.y(), p2.x(), p2.y())
        self.p1 = p1
        self.p2 = p2
        self.edge_id = edge_id
        self.selected = False
        pen = QPen(QColor(120, 120, 120), EdgeItem.WIDTH)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(pen)
        self.setZValue(1)
        self.setAcceptHoverEvents(True)

    def mousePressEvent(self, event):
        self.selected = not self.selected
        self.update_style()
        print(f"Edge clicked: id={self.edge_id}, selected={self.selected}")
        event.accept()

    def update_style(self):
        if self.selected:
            self.setPen(QPen(QColor(30, 144, 255), EdgeItem.WIDTH + 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        else:
            self.setPen(QPen(QColor(120, 120, 120), EdgeItem.WIDTH, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))

    def hoverEnterEvent(self, event):
        cur_pen = self.pen()
        self.setPen(QPen(cur_pen.color(), cur_pen.width() + 2))

    def hoverLeaveEvent(self, event):
        self.update_style()


class BoardView(QGraphicsView):
    """
    QGraphics-based Board view: draws hex tiles in rows [3,4,5,4,3], creates shared Nodes and Edges,
    and makes them clickable/highlightable.
    """
    ROW_COUNTS = [3, 4, 5, 4, 3]

    def __init__(self, parent: QWidget = None, hex_radius: float = 70):
        super().__init__(parent)
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.Antialiasing)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.hex_radius = hex_radius

        # Storage for unique nodes and edges
        # node_map: keyed by (rounded_x, rounded_y) -> (node_id, NodeItem)
        self.node_map: Dict[Tuple[int, int], Tuple[int, NodeItem]] = {}
        # edge_map: keyed by frozenset({node_id1, node_id2}) -> EdgeItem
        self.edge_map: Dict[frozenset, EdgeItem] = {}
        self._next_node_id = 0
        self._next_edge_id = 0

        self._build_board()
        self.setBackgroundBrush(QBrush(QColor(230, 230, 230)))
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMinimumSize(600, 600)
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _build_board(self):
        """
        Iterates rows and creates hexes, nodes and edges with sharing.
        """
        # compute hex width/height for pointy-top hexes
        r = self.hex_radius
        hex_w = 2 * r
        hex_h = math.sqrt(3) * r  # vertical span

        # compute overall board width to center rows
        max_tiles = max(self.ROW_COUNTS)
        # horizontal spacing between hex centers (pointy-top): r * 3/2?
        h_spacing = r * 1.75  # slightly tightened spacing so tiles touch properly
        v_spacing = hex_h * 0.88  # vertical spacing between row centers
        board_width = (max_tiles - 1) * h_spacing + hex_w

        # initial Y offset (top of the board)
        current_y = 0.0

        all_hex_centers = []

        for row_idx, tiles in enumerate(self.ROW_COUNTS):
            # row width in pixels
            row_width = (tiles - 1) * h_spacing + hex_w
            start_x = (board_width - row_width) / 2.0 + r  # center offset plus radius
            for i in range(tiles):
                cx = start_x + i * h_spacing
                cy = current_y + hex_h / 2.0
                all_hex_centers.append(QPointF(cx, cy))
            current_y += v_spacing

        # shift the entire board to positive coords and center in scene
        # find min coords
        min_x = min(pt.x() for pt in all_hex_centers)
        min_y = min(pt.y() for pt in all_hex_centers)
        # margin
        margin = 40
        translate_x = -min_x + margin
        translate_y = -min_y + margin

        # draw hexes, nodes, edges
        for center in all_hex_centers:
            center = QPointF(center.x() + translate_x, center.y() + translate_y)
            hex_item = HexItem(center, r)
            self.scene.addItem(hex_item)
            corners = self._hex_corners(center, r)
            node_ids = []
            for corner in corners:
                node_id = self._get_or_create_node(corner)
                node_ids.append(node_id)

            # create edges between consecutive corner node ids (and close the loop)
            for i in range(6):
                n1 = node_ids[i]
                n2 = node_ids[(i + 1) % 6]
                self._get_or_create_edge(n1, n2)

        # set scene rect to encompass drawn items
        self.scene.setSceneRect(self.scene.itemsBoundingRect())

    def _hex_corners(self, center: QPointF, r: float):
        pts = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.radians(angle_deg)
            x = center.x() + r * math.cos(angle_rad)
            y = center.y() + r * math.sin(angle_rad)
            pts.append(QPointF(x, y))
        return pts

    def _normalize_point(self, p: QPointF, tolerance: float = 6.0) -> Tuple[int, int]:
        """
        Normalize point coordinates to keys with rounding so adjacent hex corners coincide.
        """
        # round to nearest 'tolerance' grid to avoid floating point mismatch
        key_x = int(round(p.x() / tolerance))
        key_y = int(round(p.y() / tolerance))
        return (key_x, key_y)

    def _get_or_create_node(self, pos: QPointF) -> int:
        key = self._normalize_point(pos)
        if key in self.node_map:
            return self.node_map[key][0]
        else:
            node_id = self._next_node_id
            self._next_node_id += 1
            # compute exact position: center of bucket -> multiply back
            exact_pos = QPointF(key[0] * 6.0, key[1] * 6.0)
            # but better use the original pos for placement to reduce visual errors
            node_item = NodeItem(pos, node_id)
            self.scene.addItem(node_item)
            self.node_map[key] = (node_id, node_item)
            return node_id

    def _get_or_create_edge(self, node_id1: int, node_id2: int) -> EdgeItem:
        edge_key = frozenset({node_id1, node_id2})
        if edge_key in self.edge_map:
            return self.edge_map[edge_key]
        # resolve positions for node ids
        pos1 = self._get_node_pos(node_id1)
        pos2 = self._get_node_pos(node_id2)
        if pos1 is None or pos2 is None:
            # safety fallback (shouldn't happen)
            return None
        edge_item = EdgeItem(pos1, pos2, self._next_edge_id)
        self._next_edge_id += 1
        self.scene.addItem(edge_item)
        self.edge_map[edge_key] = edge_item
        return edge_item

    def _get_node_pos(self, node_id: int) -> QPointF:
        # search node_map values
        for (nid_key, (nid, item)) in self.node_map.items():
            if nid == node_id:
                # return center of ellipse
                rect = item.rect()
                return QPointF(rect.center().x(), rect.center().y())
        return None


# Small demo panel to run the board independently
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
