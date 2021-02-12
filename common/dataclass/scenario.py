from __future__ import annotations
import dataclasses

from common import constants, enums
from common.dataclass import settings


@dataclasses.dataclass
class Scenario:
    environment_type: enums.EnvironmentType
    comparison_type: enums.ComparisonType
    settings_list: list[settings.Settings]

    environment_parameters: dict[str, any] = dataclasses.field(default_factory=dict)

    gamma: float = constants.GAMMA

    training_episodes: int = constants.TRAINING_EPISODES
    episode_length_timeout: int = constants.EPISODE_LENGTH_TIMEOUT
    episodes_print_frequency: int = constants.EPISODES_PRINT_FREQUENCY

    performance_sample_start: int = constants.PERFORMANCE_SAMPLE_START
    performance_sample_frequency: int = constants.PERFORMANCE_SAMPLE_FREQUENCY

    runs: int = constants.RUNS
    run_print_frequency: int = constants.RUN_PRINT_FREQUENCY

    moving_average_window_size: int = constants.MOVING_AVERAGE_WINDOW_SIZE

    graph_parameters: dict[str, any] = dataclasses.field(default_factory=dict)
