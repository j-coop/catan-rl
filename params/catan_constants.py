# Board configuration
N_TILES = 19           # Total land tiles (including desert)
N_NODES = 54           # Intersections (settlements/cities)
N_EDGES = 72           # Paths (roads)

N_ADJACENT_TILES = 3
N_ADJACENT_EDGES = 6
N_ADJACENT_NODES = 6

N_PORT_NODES = 30

# Resources and tile types
RESOURCE_TYPES = ["brick", "wood", "sheep", "wheat", "ore", "desert"]
TILE_TYPE_COUNTS = {
    "brick": 3,
    "wood": 4,
    "sheep": 4,
    "wheat": 4,
    "ore": 3,
    "desert": 1
}
N_RESOURCE_TYPES = 6

PORT_TYPES = ["brick", "wood", "sheep", "wheat", "ore", "generic"]
PORT_TYPE_COUNTS = {
    "brick": 1,
    "wood": 1,
    "sheep": 1,
    "wheat": 1,
    "ore": 1,
    "generic": 2,  # port 3:1
}
N_PORT_FIELD_TYPES = 6

# Number tokens on tiles
ALL_TOKENS = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
TOKENS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
N_TOKEN_VALUES = 11

DICE_PROBABILITIES = {
    2: 1/36,
    3: 2/36,
    4: 3/36,
    5: 4/36,
    6: 5/36,
    7: 6/36,
    8: 5/36,
    9: 4/36,
    10: 3/36,
    11: 2/36,
    12: 1/36
}
MAX_PROBABILITY = 5/36

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

NUM_ROLLS = 100

# Expected number of resources gained in NUM_ROLLS rolls for best possible token setup (6, 6, 8)
BEST_EXPECTED_GAIN = MAX_PROBABILITY * NUM_ROLLS * 3

# Rewards relative weights importance
REWARD_WEIGHTS = {
    "ROAD": 2,
    "RESOURCES_NUM": 4,
    "RESOURCES_DISTRIBUTION": 4
}

# Number of steps in each episode
# (4 players place 2 settlements and 2 roads in total)
STEPS_PER_EPISODE = 16

# Agent training number of episodes
N_EPISODES = 100
