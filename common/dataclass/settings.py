from __future__ import annotations
import dataclasses
import copy
from typing import Optional

from common import enums
from common.dataclass import algorithm_parameters, policy_parameters


@dataclasses.dataclass
class Settings:
    gamma: Optional[float] = None

    algorithm_parameters: algorithm_parameters.AlgorithmParameters = \
        dataclasses.field(default_factory=algorithm_parameters.default_factory)

    policy_parameters: policy_parameters.PolicyParameters = \
        dataclasses.field(default_factory=policy_parameters.default_factory)

    runs: Optional[int] = None
    run_print_frequency: Optional[int] = None

    training_episodes: Optional[int] = None
    episode_length_timeout: Optional[int] = None
    episode_print_frequency: Optional[int] = None

    episode_to_start_recording: Optional[int] = None
    episode_recording_frequency: Optional[int] = None
    review_every_step: Optional[bool] = None

    algorithm_title: str = ""   # algorithm title will be populated by Trainer later whether it's used or not


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
    algorithm_parameters=algorithm_parameters.AlgorithmParameters(
        initial_v_value=0.0,
        initial_q_value=0.0,
        verbose=False
    ),
    policy_parameters=policy_parameters.PolicyParameters(
        policy_type=enums.PolicyType.E_GREEDY,
        epsilon=0.1
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


def default_factory() -> Settings:
    return copy.deepcopy(default_settings)
