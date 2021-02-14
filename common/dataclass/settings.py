from __future__ import annotations
import dataclasses
from typing import Optional

from common import enums
from common.dataclass import algorithm_parameters_


@dataclasses.dataclass
class Settings:
    gamma: Optional[float] = None

    algorithm_type: Optional[enums.AlgorithmType] = None
    algorithm_parameters: algorithm_parameters_.AlgorithmParameters = \
        dataclasses.field(default_factory=algorithm_parameters_.default_factory)

    runs: Optional[int] = None
    run_print_frequency: Optional[int] = None

    training_episodes: Optional[int] = None
    episode_length_timeout: Optional[int] = None
    episode_print_frequency: Optional[int] = None

    episode_to_start_recording: Optional[int] = None
    episode_recording_frequency: Optional[int] = None
    review_every_step: Optional[bool] = None

    algorithm_title: str = ""   # algorithm title will be populated by Trainer later whether it's used or not

    def test(self):
        print(default_settings)


precedence_attribute_names: list[str] = [
    'gamma',
    'runs',
    'run_print_frequency',
    'training_episodes',
    'episode_length_timeout',
    'episode_print_frequency',
    'episode_to_start_recording',
    'episode_recording_frequency',
    'review_every_step'
]

default_settings = Settings(
    gamma=1.0,
    algorithm_parameters=algorithm_parameters_.AlgorithmParameters(
        initial_v_value=0.0,
        initial_q_value=0.0,
        verbose=True
    ),
    runs=10,
    run_print_frequency=10,
    training_episodes=100,
    episode_length_timeout=10000,
    episode_print_frequency=1000,
    episode_to_start_recording=0,
    episode_recording_frequency=1,
    review_every_step=False
)
