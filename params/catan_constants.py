# Board configuration
N_TILES = 19           # Total land tiles (including desert)
N_NODES = 54           # Intersections (settlements/cities)
N_EDGES = 72           # Paths (roads)

N_ADJACENT_TILES = 3
N_ADJACENT_EDGES = 6
N_ADJACENT_NODES = 6

N_PORT_NODES = 30

# Resources and tile types
RESOURCE_TYPES = ["brick", "wood", "wool", "grain", "ore", "none"]
TILE_TYPE_COUNTS = {
    "brick": 3,
    "wood": 4,
    "wool": 4,
    "grain": 4,
    "ore": 3,
    "desert": 1  # desert
}
N_RESOURCE_TYPES = 6

PORT_TYPE_COUNTS = {
    "brick": 1,
    "wood": 1,
    "wool": 1,
    "grain": 1,
    "ore": 1,
    "generic": 4,  # port 3:1
    "no_port": 6
}
N_PORT_FIELD_TYPES = 7

# Number tokens on tiles
TOKENS = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
N_TOKEN_VALUES = 11


# Player limits
N_PLAYERS = 4
ROADS_PER_PLAYER = 15
SETTLEMENTS_PER_PLAYER = 5
CITIES_PER_PLAYER = 4

# Development cards (optional for early version)
DEV_CARD_COUNTS = {
    "knight": 14,
    "victory_point": 5,
    "road_building": 2,
    "year_of_plenty": 2,
    "monopoly": 2
}
TOTAL_DEV_CARDS = sum(DEV_CARD_COUNTS.values())

# Game phases (if useful for state management)
PHASES = ["setup", "roll_dice", "build", "trade", "end_turn"]
