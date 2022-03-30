from __future__ import annotations
from dataclasses import dataclass

from mdp import common
from mdp.model.non_tabular.feature.tile_coding.tiling_coding_parameters import TileCodingParameters
from mdp.model.non_tabular.feature.tile_coding.tiling_group_parameters import TilingGroupParameters
from mdp.task.mountain_car.comparison.comparison_builder import ComparisonBuilder
from mdp.task.mountain_car.comparison.comparison import Comparison
from mdp.task.mountain_car.enums import Dim

# TODO: build episode multiprocessing


@dataclass
class Settings(common.Settings):
    runs: int = 1
    training_episodes: int = 9000
    episode_print_frequency: int = 10
    # display_every_step: bool = True
    algorithm_parameters: common.AlgorithmParameters = common.AlgorithmParameters(
        algorithm_type=common.AlgorithmType.NON_TABULAR_EPISODIC_SARSA,
    )
    policy_parameters: common.PolicyParameters = common.PolicyParameters(
        policy_type=common.PolicyType.NON_TABULAR_E_GREEDY,
        epsilon=0.0
    )
    dual_policy_relationship: common.DualPolicyRelationship = common.DualPolicyRelationship.LINKED_POLICIES
    feature_parameters: TileCodingParameters = TileCodingParameters(
        feature_type=common.FeatureType.TILE_CODING,
        tiling_group=TilingGroupParameters(
            included_dims={Dim.POSITION, Dim.VELOCITY, Dim.ACCELERATION}
        )
    )
    value_function_parameters: common.ValueFunctionParameters = common.ValueFunctionParameters(
        value_function_type=common.ValueFunctionType.LINEAR_STATE_ACTION
    )


class MountainCarStandard(ComparisonBuilder):
    def create(self) -> Comparison:
        return Comparison(
            environment_parameters=self._environment_parameters,
            comparison_settings=Settings(),
            graph3d_values=common.Graph3DValues(
                x_label="Position",
                y_label="Velocity",
                z_label="Time to go"
            )
        )
