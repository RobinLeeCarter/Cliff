from __future__ import annotations
import dataclasses
from typing import Optional

from common import enums


@dataclasses.dataclass
class Settings:
    algorithm_type: Optional[enums.AlgorithmType] = None
    algorithm_parameters: dict[str, any] = dataclasses.field(default_factory=dict)
    algorithm_title: str = ""   # algorithm title will be populated here by Trainer whether it's used or not

    runs: Optional[int] = None
    run_print_frequency: Optional[int] = None

    training_episodes: Optional[int] = None
    episode_length_timeout: Optional[int] = None
    episode_print_frequency: Optional[int] = None

    episode_to_start_recording: Optional[int] = None
    episode_recording_frequency: Optional[int] = None
