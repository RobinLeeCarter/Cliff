from __future__ import annotations
import dataclasses
from typing import Optional

from common import enums


@dataclasses.dataclass
class Settings:
    gamma: Optional[float] = None

    # TODO: potentially a dataclass hierarchy
    algorithm_type: Optional[enums.AlgorithmType] = None
    algorithm_parameters: dict[str, any] = dataclasses.field(default_factory=dict)

    runs: Optional[int] = None
    run_print_frequency: Optional[int] = None

    training_episodes: Optional[int] = None
    episode_length_timeout: Optional[int] = None
    episode_print_frequency: Optional[int] = None

    episode_to_start_recording: Optional[int] = None
    episode_recording_frequency: Optional[int] = None
    review_every_step: Optional[bool] = None

    algorithm_title: str = ""   # algorithm title will be populated by Trainer later whether it's used or not


default_settings = Settings(
    gamma=1.0,
    algorithm_parameters={"initial_v_value": 0.0,
                          "initial_q_value": 0.0},
    runs=10,
    run_print_frequency=10,
    training_episodes=100,
    episode_length_timeout=10000,
    episode_print_frequency=1000,
    episode_to_start_recording=0,
    episode_recording_frequency=1,
    review_every_step=False
)
