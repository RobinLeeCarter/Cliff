from __future__ import annotations

import numpy as np

from common import enums


GAMMA: float = 1.0

INITIAL_V_VALUE: float = 0.0
INITIAL_Q_VALUE: float = 0.0

TRAINING_ITERATIONS: int = 500
EPISODE_LENGTH_TIMEOUT: int = 1000
EPISODES_PRINT_FREQUENCY: int = 1000

PERFORMANCE_SAMPLE_START: int = 0
PERFORMANCE_SAMPLE_FREQUENCY: int = 1

RUNS: int = 50
RUN_PRINT_FREQUENCY: int = 10
MOVING_AVERAGE_WINDOW_SIZE = 19


rng: np.random.Generator = np.random.default_rng()
COMPARISON: enums.ComparisonType = enums.ComparisonType.RETURN_BY_EPISODE
