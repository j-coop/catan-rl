#############################  Board parameters  ###############################
N_TILES = 19           # Total land tiles (including desert)
N_NODES = 54           # Intersections (settlements/cities)
N_EDGES = 72           # Paths (roads)
N_PLAYERS = 4
N_PORT_NODES = 30

RESOURCE_TYPES = ["wood", "brick", "sheep", "wheat", "ore"]

BUILD_COSTS = {
    "settlement": {"wood": 1, "brick": 1, "sheep": 1, "wheat": 1},
    "city": {"ore": 3, "wheat": 2},
    "road": {"wood": 1, "brick": 1},
    "dev_card": {"ore": 1, "wheat": 1, "sheep": 1},
}

N_ADJACENT_TILES = 3
N_ADJACENT_EDGES = 6
N_ADJACENT_NODES = 6

# Resources and tile types
TILE_TYPES = ["brick", "wood", "sheep", "wheat", "ore", "desert"]
TILE_TYPE_COUNTS = {
    "brick": 3,
    "wood": 4,
    "sheep": 4,
    "wheat": 4,
    "ore": 3,
    "desert": 1
}
N_TILE_TYPES = 6

PORT_TYPES = ["wood", "brick", "sheep", "wheat", "ore", "3for1"]
PORT_TYPE_COUNTS = {
    "brick": 1,
    "wood": 1,
    "sheep": 1,
    "wheat": 1,
    "ore": 1,
    "3for1": 2,
}
N_PORT_TYPES = 6

# Number tokens on tiles
ALL_TOKENS = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
TOKENS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
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
ROADS_PER_PLAYER = 15
SETTLEMENTS_PER_PLAYER = 5
CITIES_PER_PLAYER = 4

# Development cards (optional for early version)
DEV_CARD_TYPES = ["knight", "victory_point", "road_building",
                  "year_of_plenty", "monopoly"]
DEV_CARD_COUNTS = {
    "knight": 14,
    "victory_point": 5,
    "road_building": 2,
    "year_of_plenty": 2,
    "monopoly": 2
}
N_DEV_CARDS = sum(DEV_CARD_COUNTS.values())



#############################  Training parameters  ###############################
NUM_ROLLS = 100

# Expected number of resources gained in NUM_ROLLS rolls
# for best possible token setup (6, 6, 8)
BEST_EXPECTED_GAIN = MAX_PROBABILITY * NUM_ROLLS * 3

# Rewards relative weights importance
REWARD_WEIGHTS = {
    "ROAD": 0,
    "RESOURCES_NUM": 10,
    "RESOURCES_DISTRIBUTION": 0
}

# Number of steps in each episode
# (4 players place 2 settlements and 2 roads in total)
STEPS_PER_EPISODE = 16

# Agent training number of episodes
N_EPISODES = 500000

# Magic statistical average reward for settlements (from baseline_reward.py)
BASELINE_REWARD = 0.42962962962966283
