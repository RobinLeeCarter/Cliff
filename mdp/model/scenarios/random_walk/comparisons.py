from __future__ import annotations

from mdp import common


def episode() -> common.Comparison:
    comparison = common.Comparison(
        environment_parameters=common.EnvironmentParameters(
            environment_type=common.EnvironmentType.RANDOM_WALK,
            actions_list=common.ActionsList.NO_ACTIONS
        ),
        comparison_settings=common.Settings(
            runs=1,
            training_episodes=1000,
            policy_parameters=common.PolicyParameters(
                policy_type=common.PolicyType.NONE
            )
        ),
        breakdown_parameters=common.BreakdownParameters(
            breakdown_type=common.BreakdownType.RETURN_BY_EPISODE,
        ),
        settings_list=[
            common.Settings(
                algorithm_parameters=common.AlgorithmParameters(
                    algorithm_type=common.AlgorithmType.TD_0,
                    alpha=0.1
                )
            ),
        ],
        graph_values=common.GraphValues(
            show_graph=False
        )
    )
    return comparison
