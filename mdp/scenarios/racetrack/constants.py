import numpy as np
from mdp.scenarios.racetrack.model import tracks

MIN_VELOCITY: int = 0
MAX_VELOCITY: int = 4

MIN_ACCELERATION: int = -1
MAX_ACCELERATION: int = +1

TRACK: np.ndarray = tracks.TRACK_1

EXTRA_REWARD_FOR_FAILURE: float = -40.0   # 0.0 for problem statement

SKID_PROBABILITY: float = 0.1   # 0.1 for problem statement
