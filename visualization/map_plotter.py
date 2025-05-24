import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

from params.visualization_constants import (TILE_COLOR_MAP,
                                            HEX_RADIUS,
                                            PLAYER_COLOR_MAP)
from params.catan_constants import (RESOURCE_TYPES,
                                    N_NODES)
from params.tiles2nodes_adjacency_map import TILES_TO_NODES


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


class CatanMapPlotter:

    def __init__(self, base_obs):
        self.__resources = base_obs['tiles']['resources']
        self.__tokens = base_obs['tiles']['tokens']
        self.__nodes = base_obs['nodes']

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
            self.__plot_hex(x, y, color, token)

    def __plot_sea_hexes(self):
        for q, r in SEA_POSITIONS:
            x, y = self.__get_hex_position(q, r)
            self.__plot_hex(x, y, TILE_COLOR_MAP['water'])

    def __plot_hex(self, x, y, color, token=0):
        hex = patches.RegularPolygon((x, y),
                                    numVertices=6,
                                    radius=HEX_RADIUS * 0.95,
                                    orientation=np.radians(60),
                                    facecolor=color,
                                    edgecolor='black')
        self.__ax.add_patch(hex)
        if token != 0:
            self.__ax.text(x, y, str(token),
                    ha='center', 
                    va='center',
                    fontsize=15,
                    color='black')
            
    def __plot_settlements(self):
        plotted_nodes = np.zeros((N_NODES,), dtype=np.int8)
        angles = [math.radians(a) for a in [150, 90, 30, 210, 270, 330]]

        for i, (q, r) in enumerate(LAND_POSITIONS):
            for k in range(6):
                if plotted_nodes[TILES_TO_NODES[i][k]] == 0:
                    x_center, y_center = self.__get_hex_position(q, r)
                    angle = angles[k]
                    x_node = x_center + HEX_RADIUS * math.cos(angle)
                    y_node = y_center + HEX_RADIUS * math.sin(angle)
                    owners_vec = self.__nodes["owner"][i][k]
                    player_id = None
                    if np.max(owners_vec) == 1:
                        player_id = np.argmax(owners_vec)
                    if player_id is not None:
                        self.__ax.plot(x_node, y_node,
                                    marker='D', 
                                    color=PLAYER_COLOR_MAP[player_id],
                                    markersize=16,
                                    markeredgewidth=2,
                                    markeredgecolor='black')
                    plotted_nodes[TILES_TO_NODES[i][k]] = 1

    def __get_hex_position(self, q, r):
        x = HEX_RADIUS * math.sqrt(3) * (q + r/2)
        y = HEX_RADIUS * 3/2 * r
        return x, y

    def plot_catan_map(self):
        self.__setup_plot_area()
        self.__plot_land_hexes()
        self.__plot_sea_hexes()
        self.__plot_settlements()
        plt.show()

num_tiles = 19
num_resources = 5
num_tokens = 12  # Catan uses numbers from 2 to 12 (excluding 7)

# Generate dummy resource one-hot arrays
resources = np.eye(num_resources)[np.random.choice(num_resources, num_tiles)]

# Generate dummy tokens (skipping 7)
possible_tokens = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
tokens = np.eye(13)[np.random.choice(possible_tokens, num_tiles)]

nodes = np.eye(13)[np.random.choice(possible_tokens, num_tiles)]

owners = np.zeros((19, 6, 4), dtype=int)

# Simulate some ownership (e.g., player 0 owns node 2 on tile 5)
owners[5][4][3] = 1
owners[1][4][1] = 1

owners[16][3][0] = 1
owners[18][5][2] = 1

# Create base_obs input
base_obs = {
    'tiles': {
        'resources': resources,
        'tokens': tokens
    },
    'nodes': {
        "owner": owners
    }
}

# Create plotter instance and plot the map
plotter = CatanMapPlotter(base_obs)
plotter.plot_catan_map()