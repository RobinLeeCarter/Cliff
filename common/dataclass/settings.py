from __future__ import annotations
import dataclasses

from common import constants, enums


@dataclasses.dataclass
class Settings:
    algorithm_type: enums.AlgorithmType
    parameters: dict[str, any]

    algorithm_title: str = ""   # algorithm title will be populated here by Trainer whether it's used or not

    training_iterations: int = constants.TRAINING_ITERATIONS
    episode_length_timeout: int = constants.EPISODE_LENGTH_TIMEOUT
    iteration_print_frequency: int = constants.ITERATION_PRINT_FREQUENCY

    performance_sample_start: int = constants.PERFORMANCE_SAMPLE_START
    performance_sample_frequency: int = constants.PERFORMANCE_SAMPLE_FREQUENCY

    runs: int = constants.RUNS
    run_print_frequency: int = constants.RUN_PRINT_FREQUENCY

    moving_average_window_size: int = constants.MOVING_AVERAGE_WINDOW_SIZE
