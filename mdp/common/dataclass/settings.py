from __future__ import annotations
import dataclasses
import copy
from typing import Optional

from mdp.common.dataclass import algorithm_parameters, policy_parameters


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
    display_every_step: Optional[bool] = None

    # algorithm title will be populated by Trainer later whether it's used or not
    algorithm_title: str = dataclasses.field(default="", init=False)

    def apply_default_to_nones(self, default_: Settings):
        attribute_names: list[str] = [
            'gamma',
            'runs',
            'run_print_frequency',
            'training_episodes',
            'episode_length_timeout',
            'episode_print_frequency',
            'episode_to_start_recording',
            'episode_recording_frequency',
            'review_every_step',
            'display_every_step'
        ]
        for attribute_name in attribute_names:
            attribute = self.__getattribute__(attribute_name)
            if attribute is None:
                default_value = default_.__getattribute__(attribute_name)
                self.__setattr__(attribute_name, default_value)

        self.algorithm_parameters.apply_default_to_nones(default_.algorithm_parameters)
        self.policy_parameters.apply_default_to_nones(default_.policy_parameters)


default = Settings(
    gamma=1.0,
    algorithm_parameters=algorithm_parameters.default,
    policy_parameters=policy_parameters.default,
    runs=10,
    run_print_frequency=10,
    training_episodes=100,
    episode_length_timeout=10000,
    episode_print_frequency=1000,
    episode_to_start_recording=0,
    episode_recording_frequency=1,
    review_every_step=False,
    display_every_step=False
)


def default_factory() -> Settings:
    return copy.deepcopy(default)
