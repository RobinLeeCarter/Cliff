from __future__ import annotations
from dataclasses import dataclass

from mdp import common
from mdp.task.mountain_car.comparison.comparison_builder import ComparisonBuilder
from mdp.task.mountain_car.comparison.comparison import Comparison
from mdp.task.mountain_car.model.environment_parameters import EnvironmentParameters


@dataclass
class Settings(common.Settings):
    runs: int = 1
    training_episodes: int = 1
    episode_print_frequency: int = 1
    # display_every_step: bool = True
    algorithm_parameters: common.AlgorithmParameters = common.AlgorithmParameters(
        algorithm_type=common.AlgorithmType.NON_TABULAR_EPISODIC_SARSA,
        initial_q_value=0.0,
    )
    policy_parameters: common.PolicyParameters = common.PolicyParameters(
        policy_type=common.PolicyType.NON_TABULAR_E_GREEDY,
        epsilon=0.0
    )
    dual_policy_relationship: common.DualPolicyRelationship = common.DualPolicyRelationship.LINKED_POLICIES


class MountainCarStandard(ComparisonBuilder):
    def create(self):
        return Comparison(
            environment_parameters=EnvironmentParameters(),
            comparison_settings=Settings(),
        )
