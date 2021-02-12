from __future__ import annotations

import numpy as np

from common import dataclass

rng: np.random.Generator = np.random.default_rng()
GAMMA: float = 1.0
MOVING_AVERAGE_WINDOW_SIZE = 19

# INITIAL_V_VALUE: float = 0.0
# INITIAL_Q_VALUE: float = 0.0

default_settings = dataclass.Settings(
    algorithm_parameters={"initial_v_value": 0.0,
                          "initial_q_value": 0.0},
    runs=50,
    run_print_frequency=10,
    training_episodes=100,
    episode_length_timeout=1000,
    episode_print_frequency=1000,
    episode_to_start_recording=0,
    episode_recording_frequency=1
)
