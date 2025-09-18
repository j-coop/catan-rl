N_NODES = 54
N_EDGES = 72
N_PLAYERS = 4

RESOURCE_TYPES = ["wood", "brick", "sheep", "wheat", "ore"]
DEV_CARD_TYPES = ["knight", "victory_point", "road_building", "year_of_plenty", "monopoly"]
PORT_TYPES = ["wood", "brick", "sheep", "wheat", "ore", "3for1"]

BUILD_COSTS = {
    "settlement": {"wood": 1, "brick": 1, "sheep": 1, "wheat": 1},
    "city": {"ore": 3, "wheat": 2},
    "road": {"wood": 1, "brick": 1},
    "dev_card": {"ore": 1, "wheat": 1, "sheep": 1},
}

