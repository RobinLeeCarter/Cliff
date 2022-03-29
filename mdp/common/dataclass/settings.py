from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from mdp.common.enums import DualPolicyRelationship, ParallelContextType
from mdp.common.dataclass.algorithm_parameters import AlgorithmParameters
from mdp.common.dataclass.policy_parameters import PolicyParameters
from mdp.common.dataclass.feature_parameters import FeatureParameters
from mdp.common.dataclass.value_function_parameters import ValueFunctionParameters
from mdp.common.dataclass.result_parameters import ResultParameters


@dataclass
class Settings:
    gamma: float = 1.0

    # defaults are set in set_none_to_default
    algorithm_parameters: AlgorithmParameters = field(default_factory=AlgorithmParameters)
    policy_parameters: PolicyParameters = field(default_factory=PolicyParameters)
    behaviour_policy_parameters: Optional[PolicyParameters] = None
    dual_policy_relationship: DualPolicyRelationship = DualPolicyRelationship.SINGLE_POLICY

    # non-tabular parameters
    feature_parameters: Optional[FeatureParameters] = None
    value_function_parameters: Optional[ValueFunctionParameters] = None

    runs: int = 10
    runs_multiprocessing: Optional[ParallelContextType] = None
    run_print_frequency: int = 10

    training_episodes: int = 100
    episode_length_timeout: int = 10000
    episode_print_frequency: int = 1000

    episode_to_start_recording: int = 0
    episode_recording_frequency: int = 1
    review_every_step: bool = False
    display_every_step: bool = False

    # only used for parallel routines to determine what Trainer should return from the child process
    result_parameters: Optional[ResultParameters] = None
    # result_parameters: result_parameters_.ResultParameters = \
    #     field(default_factory=result_parameters_.none_factory)

