from __future__ import annotations
import dataclasses
import copy
from typing import Optional

import utils
from mdp.common import enums
from mdp.common.dataclass import algorithm_parameters_, policy_parameters_, result_parameters_


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
    dual_policy_relationship: Optional[enums.DualPolicyRelationship] = None

    runs: Optional[int] = None
    runs_multiprocessing: Optional[enums.ParallelContextType] = None
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

    # only used for parallel routines to determine what Trainer should return from the child process
    result_parameters: Optional[result_parameters_.ResultParameters] = None
    # result_parameters: result_parameters_.ResultParameters = \
    #     dataclasses.field(default_factory=result_parameters_.none_factory)

    def set_none_to_default(self, default_: Settings):
        utils.set_none_to_default(self, default_)
        utils.set_none_to_default(self.algorithm_parameters, default_.algorithm_parameters)
        utils.set_none_to_default(self.policy_parameters, default_.policy_parameters)
        # utils.set_none_to_default(self.result_parameters, default_.result_parameters)


default = Settings(
    gamma=1.0,
    algorithm_parameters=algorithm_parameters_.default,
    policy_parameters=policy_parameters_.default,
    dual_policy_relationship=enums.DualPolicyRelationship.SINGLE_POLICY,
    runs=10,
    run_print_frequency=10,
    runs_multiprocessing=enums.ParallelContextType.NONE,
    training_episodes=100,
    episode_length_timeout=10000,
    episode_print_frequency=1000,
    episode_to_start_recording=0,
    episode_recording_frequency=1,
    review_every_step=False,
    display_every_step=False,
    # result_parameters=None,
)


def default_factory() -> Settings:
    return copy.deepcopy(default)
