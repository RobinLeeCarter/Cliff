from __future__ import annotations
import dataclasses
from typing import Optional

from common import enums


@dataclasses.dataclass
class Settings:
    gamma: Optional[float] = None

    algorithm_type: Optional[enums.AlgorithmType] = None
    algorithm_parameters: dict[str, any] = dataclasses.field(default_factory=dict)

    runs: Optional[int] = None
    run_print_frequency: Optional[int] = None

    training_episodes: Optional[int] = None
    episode_length_timeout: Optional[int] = None
    episode_print_frequency: Optional[int] = None

    episode_to_start_recording: Optional[int] = None
    episode_recording_frequency: Optional[int] = None

    algorithm_title: str = ""   # algorithm title will be populated by Trainer later whether it's used or not
