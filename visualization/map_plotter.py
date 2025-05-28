import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

from params.visualization_constants import (TILE_COLOR_MAP,
                                            HEX_RADIUS,
                                            PLAYER_COLOR_MAP)
from params.catan_constants import (RESOURCE_TYPES,
                                    N_NODES,
                                    N_EDGES,
                                    PORT_TYPES)
from params.tiles2nodes_adjacency_map import TILES_TO_NODES
from params.edges_list import EDGES_LIST


SEA_POSITIONS = [
    (-3, 1), (-3, 2), (-3, 3), (-2, 3), (-1, 3), (0, 3),
    (2, 1), (3, -1), (3, -2), (2, -3), (1, -3), (0, -3),
    (-1, -2), (-2, -1), (-3, 0), (3, 0), (3, -3), (1, 2),
]

LAND_POSITIONS = [
            (-2, 2), (-1, 2), (0, 2),         # top row
        (-2, 1), (-1, 1), (0, 1), (1, 1),     # next row
    (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), # middle row
        (-1, -1), (0, -1), (1, -1), (2, -1),  # next row
            (0, -2), (1, -2), (2, -2)         # bottom row
]

ANGLES = np.radians([150, 90, 30, 330, 270, 210])


class CatanMapPlotter:

    def __init__(self, base_obs):
        self.__resources = base_obs['tiles']['resources']
        self.__tokens = base_obs['tiles']['tokens']
        self.__nodes = base_obs['tiles']['nodes']
        self.__edges = base_obs['tiles']['edges']

    def __setup_plot_area(self):
        # Create a wide figure (18:9) with constrained map region
        _, ax = plt.subplots(figsize=(18, 9))
        try:
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
        except:
            pass
        ax.set_aspect('equal')

        # Left-align the map by adjusting axis limits
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.axis('off')

        # Shrink plot to left 60% of screen
        ax.set_position([-0.05, 0.05, 0.6, 0.9]) # [left, bottom, width, height]
        self.__ax = ax

    def __plot_land_hexes(self):
        for i, (q, r) in enumerate(LAND_POSITIONS):
            x, y = self.__get_hex_position(q, r)
            resource_type = np.argmax(self.__resources[i])
            color_name = RESOURCE_TYPES[resource_type]
            color = TILE_COLOR_MAP[color_name]
            token = np.argmax(self.__tokens[i])
            if token == 0 and np.sum(self.__tokens[i]) == 0:
                token = -1
            self.__plot_hex(x, y, color, token)

    def __plot_sea_hexes(self):
        for q, r in SEA_POSITIONS:
            x, y = self.__get_hex_position(q, r)
            self.__plot_hex(x, y, TILE_COLOR_MAP['water'])

    def __plot_hex(self, x, y, color, token=-1):
        hex = patches.RegularPolygon((x, y),
                                    numVertices=6,
                                    radius=HEX_RADIUS * 0.95,
                                    orientation=np.radians(60),
                                    facecolor=color,
                                    edgecolor='black')
        self.__ax.add_patch(hex)
        if token >= 0:
            self.__ax.text(x, y, str(token + 2),
                    ha='center', 
                    va='center',
                    fontsize=15,
                    color='black')
            
    def __plot_settlements(self):
        plotted_nodes = np.zeros((N_NODES,), dtype=np.int8)
        for i, (q, r) in enumerate(LAND_POSITIONS):
            nodes = TILES_TO_NODES[i][:3] + list(reversed(TILES_TO_NODES[i][3:]))
            x_tile_center, y_tile_center = self.__get_hex_position(q, r)
            for k in range(6):
                if plotted_nodes[nodes[k]] == 0:
                    owners_vec = self.__nodes["owner"][i][k]
                    if np.any(owners_vec):
                        player_id = np.argmax(owners_vec)
                        x_node = x_tile_center + HEX_RADIUS * math.cos(ANGLES[k])
                        y_node = y_tile_center + HEX_RADIUS * math.sin(ANGLES[k])
                        self.__ax.plot(x_node, y_node,
                                    marker='D', 
                                    color=PLAYER_COLOR_MAP[player_id],
                                    markersize=16,
                                    markeredgewidth=2,
                                    markeredgecolor='black')
                plotted_nodes[nodes[k]] = 1

    def __plot_roads(self):
        plotted_edges = np.zeros((N_EDGES,), dtype=np.int8)
        for i, (q, r) in enumerate(LAND_POSITIONS):
            nodes = TILES_TO_NODES[i][:3] + list(reversed(TILES_TO_NODES[i][3:]))
            x_tile_center, y_tile_center = self.__get_hex_position(q, r)
            for k in range(6):
                node_a = nodes[k]
                node_b = nodes[(k + 1) % 6]
                edge_key = tuple(sorted((node_a, node_b)))
                edge_id = EDGES_LIST.index(edge_key)
                if edge_id == -1:
                    raise ValueError(
                        "Action must specify either 1 road or 1 settlement.")
                if plotted_edges[edge_id] == 0:
                    edge_owners = self.__edges["owner"][i][k]
                    if np.any(edge_owners):
                        x_a = x_tile_center + HEX_RADIUS * math.cos(ANGLES[k])
                        y_a = y_tile_center + HEX_RADIUS * math.sin(ANGLES[k])
                        x_b = x_tile_center + HEX_RADIUS \
                            * math.cos(ANGLES[(k + 1) % 6])
                        y_b = y_tile_center + HEX_RADIUS \
                            * math.sin(ANGLES[(k + 1) % 6])
                        player_id = np.argmax(edge_owners)
                        color = PLAYER_COLOR_MAP[player_id]
                        self.__plot_road_between_nodes_scaled(x_a, y_a,
                                                              x_b, y_b,
                                                              color)
                    plotted_edges[edge_id] = 1

    def __plot_road_between_nodes_scaled(self, x_a, y_a, x_b, y_b, 
                                         color,
                                         linewidth=5,
                                         scale=0.55):
        # Vector from A to B
        dx, dy = (x_b - x_a), (y_b - y_a)

        # Midpoint
        mx, my = (x_a + x_b) / 2, (y_a + y_b) / 2

        # Half-length vector scaled by scale/2
        half_dx, half_dy = dx * scale / 2, dy * scale / 2

        # New endpoints, centered but shorter
        new_x_a, new_y_a = mx - half_dx, my - half_dy
        new_x_b, new_y_b = mx + half_dx, my + half_dy

        # Draw black edge line (thicker)
        self.__ax.plot(
            [new_x_a, new_x_b],
            [new_y_a, new_y_b],
            color='black',
            linewidth=linewidth + 2 * 2,
            solid_capstyle='round',
            zorder=1,
        )

        self.__ax.plot(
            [new_x_a, new_x_b],
            [new_y_a, new_y_b],
            color=color,
            linewidth=linewidth,
            solid_capstyle='round',
            zorder=2,
        )

    def __plot_ports(self):
        drawn_ports = set()
        for tile_index, (q, r) in enumerate(LAND_POSITIONS):
            x_tile, y_tile = self.__get_hex_position(q, r)

            for node_index in range(6):
                port_vec = self.__nodes["ports"][tile_index][node_index]
                if not np.any(port_vec):
                    continue

                port_type_index = np.argmax(port_vec)
                port_type = PORT_TYPES[port_type_index]

                # Check if the adjacent node also has the same port
                next_index = (node_index + 1) % 6
                next_port_vec = self.__nodes["ports"][tile_index][next_index]

                is_pair = (
                    np.any(next_port_vec)
                    and np.argmax(next_port_vec) == port_type_index
                )

                if is_pair:
                    # Only draw once per pair
                    if (tile_index, port_type_index) in drawn_ports:
                        continue
                    drawn_ports.add((tile_index, port_type_index))

                    angle_a = ANGLES[node_index]
                    angle_b = ANGLES[next_index]
                    avg_angle = (angle_a + angle_b) / 2

                    # Get anchor position slightly off the tile
                    offset = HEX_RADIUS * 1.8
                    x_anchor = x_tile + offset * math.cos(avg_angle)
                    y_anchor = y_tile + offset * math.sin(avg_angle)

                    # Draw anchor symbol
                    self.__ax.text(
                        x_anchor, y_anchor, "⚓",
                        ha='center', va='center',
                        fontsize=20,
                        fontweight='bold',
                        zorder=10
                    )

                    # Label: either resource or '3:1'
                    label = port_type if port_type != 'generic' else '3:1'
                    self.__ax.text(
                        x_anchor, y_anchor - 0.2,
                        label,
                        ha='center', va='top',
                        fontsize=10,
                        fontweight='bold',
                        color='black',
                        zorder=11
                    )

                    # Draw lines to both involved nodes
                    for angle in [angle_a, angle_b]:
                        x_node = x_tile + HEX_RADIUS * math.cos(angle)
                        y_node = y_tile + HEX_RADIUS * math.sin(angle)
                        self.__ax.plot(
                            [x_node, x_anchor], [y_node, y_anchor],
                            color='black',
                            linewidth=2,
                            linestyle='dotted',
                            zorder=9
                        )

    def __get_hex_position(self, q, r):
        x = HEX_RADIUS * math.sqrt(3) * (q + r/2)
        y = HEX_RADIUS * 3/2 * r
        return x, y

    def plot_catan_map(self):
        self.__setup_plot_area()
        self.__plot_land_hexes()
        self.__plot_sea_hexes()
        self.__plot_settlements()
        self.__plot_roads()
        self.__plot_ports()
        plt.show()
