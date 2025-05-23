import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

from params.visualization_constants import (TILE_COLOR_MAP,
                                            HEX_RADIUS)
from params.catan_constants import RESOURCE_TYPES


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
    
    def __get_hex_position(self, q, r):
        x = HEX_RADIUS * math.sqrt(3) * (q + r/2)
        y = HEX_RADIUS * 3/2 * r
        return x, y

    def plot_catan_map(self):
        self.__setup_plot_area()
        self.__plot_land_hexes()
        self.__plot_sea_hexes()
        plt.show()
