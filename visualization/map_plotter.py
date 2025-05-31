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
                                    PORT_TYPES,
                                    N_TILES)
from params.tiles2nodes_adjacency_map import TILES_TO_NODES
from params.tiles2edges_adjacency_map import TILES_TO_EDGES
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
        for node_id in range(N_NODES):
            for tile_id in range(N_TILES):
                if node_id in TILES_TO_NODES[tile_id]:
                    if plotted_nodes[node_id] == 0:
                        index = TILES_TO_NODES[tile_id].index(node_id)
                        owners_vec = self.__nodes["owner"][tile_id][index]
                        if np.any(owners_vec):
                            player_id = np.argmax(owners_vec)
                            self.__plot_settlement(node_id, player_id)
                        plotted_nodes[node_id] = 1
                        break
                    else:
                        break

    def __plot_settlement(self, node_id, player_id):
        for tile_id, (q, r) in enumerate(LAND_POSITIONS):
            if node_id in TILES_TO_NODES[tile_id]:
                k = TILES_TO_NODES[tile_id].index(node_id)
                x_tile_center, y_tile_center = self.__get_hex_position(q, r)
                x_node = x_tile_center + HEX_RADIUS * math.cos(ANGLES[k])
                y_node = y_tile_center + HEX_RADIUS * math.sin(ANGLES[k])
                self.__plot_settlement_marker(x_node, y_node, player_id)
                break

    def __plot_settlement_marker(self, x_node, y_node, player_id):
        self.__ax.plot(x_node, y_node,
                       marker='D', 
                       color=PLAYER_COLOR_MAP[player_id],
                       markersize=16,
                       markeredgewidth=2,
                       markeredgecolor='black')
        
    def __plot_city_marker(self, x_node, y_node, player_id):
        # TODO Finish this func basing on the above one
        pass

    def __plot_roads(self):
        plotted_edges = np.zeros((N_EDGES,), dtype=np.int8)
        for edge_id in range(len(EDGES_LIST)):
            edge = EDGES_LIST[edge_id]
            for tile_id in range(N_TILES):
                if edge in TILES_TO_EDGES[tile_id]:
                    if plotted_edges[edge_id] == 0:
                        index = TILES_TO_EDGES[tile_id].index(edge)
                        owners_vec = self.__edges["owner"][tile_id][index]
                        if np.any(owners_vec):
                            player_id = np.argmax(owners_vec)
                            self.__plot_settlement(edge, player_id)
                        plotted_edges[edge_id] = 1
                        break
                    else:
                        break

    def __plot_road(self, edge, player_id):
        for tile_id, (q, r) in enumerate(LAND_POSITIONS):
            edge = tuple(sorted(edge))
            if edge in TILES_TO_EDGES[tile_id]:
                k = TILES_TO_EDGES[tile_id].index(edge)
                x_tile_center, y_tile_center = self.__get_hex_position(q, r)
                x_a = x_tile_center + HEX_RADIUS * math.cos(ANGLES[k])
                y_a = y_tile_center + HEX_RADIUS * math.sin(ANGLES[k])
                x_b = x_tile_center + HEX_RADIUS \
                    * math.cos(ANGLES[(k + 1) % 6])
                y_b = y_tile_center + HEX_RADIUS \
                    * math.sin(ANGLES[(k + 1) % 6])
                self.__plot_road_marker(vertex_1_coords=(x_a, y_a),
                                        vertex_2_coords=(x_b, y_b),
                                        player_id=player_id)
                break

    def __plot_road_marker(self, vertex_1_coords, vertex_2_coords, player_id):
        color = PLAYER_COLOR_MAP[player_id]
        linewidth = 5
        scale = 0.55

        dx = vertex_2_coords[0] - vertex_1_coords[0]
        dy = vertex_2_coords[1] - vertex_1_coords[1]
        mx = (vertex_1_coords[0] + vertex_2_coords[0]) / 2
        my = (vertex_1_coords[1] + vertex_2_coords[1]) / 2
        half_dx, half_dy = dx * scale / 2, dy * scale / 2
        new_x_a, new_y_a = mx - half_dx, my - half_dy
        new_x_b, new_y_b = mx + half_dx, my + half_dy

        # Draw black edge line
        self.__ax.plot(
            [new_x_a, new_x_b],
            [new_y_a, new_y_b],
            color='black',
            linewidth=(linewidth + 2 * 2),
            solid_capstyle='round',
            zorder=1,
        )
        # Draw color line inside the black one
        self.__ax.plot(
            [new_x_a, new_x_b],
            [new_y_a, new_y_b],
            color=color,
            linewidth=linewidth,
            solid_capstyle='round',
            zorder=2,
        )

    def __plot_ports(self):
        for tile_index, (q, r) in enumerate(LAND_POSITIONS):
            if not np.any(self.__nodes["ports"][tile_index]):
                continue
            for node_index in range(6):
                port_vec = self.__nodes["ports"][tile_index][node_index]
                if not np.any(port_vec):
                    continue

                # Check if the adjacent node also has the same port
                next_index = (node_index + 1) % 6
                next_port_vec = self.__nodes["ports"][tile_index][next_index]
                port_type = np.argmax(port_vec)
                port_exists = (
                    np.any(next_port_vec)
                    and np.argmax(next_port_vec) == port_type
                )
                if port_exists:
                    self.__plot_port(tile_coords=(q, r),
                                     indices=(node_index, next_index),
                                     port_type_id=port_type)
                    

    def __plot_port(self, tile_coords, indices, port_type_id):
        right_edge_tiles = [(0, 2), (1, 1), (2, 0), (2, -1), (2, -2)]
        angles = (ANGLES[indices[0]], ANGLES[indices[1]])
        avg_angle = (angles[0] + angles[1]) / 2
        if indices == (2,3) and tile_coords in  right_edge_tiles:
            avg_angle = 0

        offset = HEX_RADIUS * 1.8
        x_tile, y_tile = self.__get_hex_position(tile_coords[0], tile_coords[1])
        x = x_tile + offset * math.cos(avg_angle)
        y = y_tile + offset * math.sin(avg_angle)
        port_type = PORT_TYPES[port_type_id]
        self.__plot_port_marker_with_label(x_anchor=x,
                                            y_anchor=y,
                                            port_type=port_type)
        self.__plot_lines_to_port_marker(angles=angles,
                                            tile_coords=(x_tile, y_tile),
                                            anchor_coords=(x, y))

    def __plot_port_marker_with_label(self, x_anchor, y_anchor, port_type):
        # Draw anchor symbol
        self.__ax.text(
            x_anchor, y_anchor, "⚓",
            ha='center', va='center',
            fontsize=25,
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
    
    def __plot_lines_to_port_marker(self, angles, tile_coords, anchor_coords):
        # Draw lines to both involved nodes
        for angle in angles:
            x_node = tile_coords[0] + HEX_RADIUS * math.cos(angle)
            y_node = tile_coords[1] + HEX_RADIUS * math.sin(angle)
            self.__ax.plot(
                [x_node, anchor_coords[0]], [y_node, anchor_coords[1]],
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
