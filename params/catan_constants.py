#############################  Board parameters  ###############################
N_TILES = 19           # Total land tiles (including desert)
N_NODES = 54           # Intersections (settlements/cities)
N_EDGES = 72           # Paths (roads)
N_PLAYERS = 4
N_PORT_NODES = 30

RESOURCE_TYPES = ["wood", "brick", "sheep", "wheat", "ore"]
N_RESOURCE_TYPES = 5

# Explicitly ordered 20 possible bank trade pairs
BANK_TRADE_PAIRS = [
    ("wood", "brick"),
    ("wood", "sheep"),
    ("wood", "wheat"),
    ("wood", "ore"),

    ("brick", "wood"),
    ("brick", "sheep"),
    ("brick", "wheat"),
    ("brick", "ore"),

    ("sheep", "wood"),
    ("sheep", "brick"),
    ("sheep", "wheat"),
    ("sheep", "ore"),

    ("wheat", "wood"),
    ("wheat", "brick"),
    ("wheat", "sheep"),
    ("wheat", "ore"),

    ("ore", "wood"),
    ("ore", "brick"),
    ("ore", "sheep"),
    ("ore", "wheat"),
]

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
TILE_TYPES = ["wood", "brick", "sheep", "wheat", "ore", "desert"]
TILE_TYPE_COUNTS = {
    "brick": 3,
    "wood": 4,
    "sheep": 4,
    "wheat": 4,
    "ore": 3,
    "desert": 1
}
N_TILE_TYPES = 6
MAX_RESOURCE_COUNT = 19

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
MAX_VICTORY_POINTS = 10
MAX_KNIGHTS = 14

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

LONGEST_ROAD_MIN_LENGTH = 5


# Predefined node indices that correspond to the 9 physical port locations.
# Each tuple is a pair of node indices representing the two nodes that touch that port.
PORT_NODE_PAIRS = [
    (0, 1),
    (3, 4),
    (14, 15),
    (26, 37),
    (45, 46),
    (50, 51),
    (47, 48),
    (28, 38),
    (7, 17)
]

GAMMA = 0.99

# Reward shaping schedule and terminal rewards
SHAPING_WEIGHT_START = 1.0
SHAPING_WEIGHT_END = 0.2
SHAPING_ANNEAL_STEPS = 5_000_000
WIN_REWARD = 200.0
LOSS_PENALTY = -50.0


##################  Init-placement agent training parameters  ##################
NUM_ROLLS = 100

# Expected number of resources gained in NUM_ROLLS rolls
# for best possible token setup (6, 6, 8)
BEST_EXPECTED_GAIN = MAX_PROBABILITY * NUM_ROLLS * 3

# Rewards relative weights importance
REWARD_WEIGHTS = {
    "ROAD": 1,
    "RESOURCES_NUM": 7,
    "PLACEMENT": 2,
    "RESOURCES_DISTRIBUTION": 10
}

DIVERSITY_SCORE_WEIGHT = 0.4
COVERAGE_SCORE_WEIGHT = 0.6

# Number of steps in each episode
# (4 players place 2 settlements and 2 roads in total)
INIT_PLACEMENT_ENV_STEPS_PER_EPISODE = 16

INIT_PLACEMENT_ENV_N_EPISODES = 250000
INIT_PLACEMENT_ENV_N_TIMESTEPS = INIT_PLACEMENT_ENV_N_EPISODES * INIT_PLACEMENT_ENV_STEPS_PER_EPISODE
INIT_PLACEMENT_ENV_EVAL_FREQ = 2000 * INIT_PLACEMENT_ENV_STEPS_PER_EPISODE
INIT_PLACEMENT_ENV_CHECKPOINT_SAVE_FREQ = 25000 * INIT_PLACEMENT_ENV_STEPS_PER_EPISODE
INIT_PLACEMENT_ENV_PATIENCE = 14

## Magic statistical average reward for settlements (from baseline_reward.py) ##
BASELINE_REWARD = 0.42962962962966283

VERBOSE = False

######################  Action space parameters  ###############################
SELF_SPACE_SIZE = 23
OTHERS_SPACE_SIZE = 42
BOARD_SPACE_SIZE = 1214
FULL_ACTION_SPACE_SIZE = SELF_SPACE_SIZE + OTHERS_SPACE_SIZE + BOARD_SPACE_SIZE

IS_ENCODER_ENABLED = True
BOARD_LATENT = 340
SELF_LATENT = 342
OTHERS_LATENT = 342
FINAL_LATENT = OTHERS_LATENT + SELF_LATENT + BOARD_LATENT
