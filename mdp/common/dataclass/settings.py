from __future__ import annotations
import dataclasses
import copy
from typing import Optional

from mdp.common import utils
from mdp.common.dataclass import algorithm_parameters_, policy_parameters_


@dataclasses.dataclass
class Settings:
    gamma: Optional[float] = None

    # defaults are set in set_none_to_default
    algorithm_parameters: algorithm_parameters_.AlgorithmParameters = \
        dataclasses.field(default_factory=algorithm_parameters_.none_factory)
    policy_parameters: policy_parameters_.PolicyParameters = \
        dataclasses.field(default_factory=policy_parameters_.none_factory)
    behaviour_policy_parameters: policy_parameters_.PolicyParameters = \
        dataclasses.field(default_factory=policy_parameters_.none_factory)

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

    def set_none_to_default(self, default_: Settings):
        utils.set_none_to_default(self, default_)
        utils.set_none_to_default(self.algorithm_parameters, default_.algorithm_parameters)
        utils.set_none_to_default(self.policy_parameters, default_.policy_parameters)


default = Settings(
    gamma=1.0,
    algorithm_parameters=algorithm_parameters_.default,
    policy_parameters=policy_parameters_.default,
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
