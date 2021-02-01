from data import grids
from environment import grid

GRID: grid.Grid = grids.GRID_1

ALPHA: float = 0.5
INITIAL_Q_VALUE: float = 0.0
# for deletion
EXTRA_REWARD_FOR_FAILURE: float = 0.0
SKID_PROBABILITY: float = 0.0

LEARNING_EPISODES: int = 10_000
PERFORMANCE_SAMPLE_START: int = 1000
PERFORMANCE_SAMPLE_FREQUENCY: int = 1000
PERFORMANCE_SAMPLES: int = 100
