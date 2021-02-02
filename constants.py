from data import grids, settings_list
from environment import grid

GAMMA: float = 1.0

ALPHA: float = 0.5
INITIAL_Q_VALUE: float = 0.0
# for deletion
EXTRA_REWARD_FOR_FAILURE: float = 0.0
SKID_PROBABILITY: float = 0.0

LEARNING_EPISODES: int = 500
EPISODE_LENGTH_TIMEOUT: int = 10_000

PERFORMANCE_SAMPLE_START: int = 0
PERFORMANCE_SAMPLE_FREQUENCY: int = 1
PERFORMANCE_SAMPLES: int = 500

GRID: grid.Grid = grids.GRID_1
SETTINGS_LIST = settings_list.SETTINGS_LIST
