from __future__ import annotations

from mdp import common


def episode() -> common.Comparison:
    comparison = common.Comparison(
        environment_parameters=common.EnvironmentParameters(
            environment_type=common.EnvironmentType.RANDOM_WALK,
            actions_list=common.ActionsList.NO_ACTIONS,
        ),
        comparison_settings=common.Settings(
            runs=10,
            training_episodes=100,
            policy_parameters=common.PolicyParameters(
                policy_type=common.PolicyType.NONE
            ),
            algorithm_parameters=common.AlgorithmParameters(
                initial_v_value=0.5
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
            show_graph=True
        ),
        grid_view_parameters=common.GridViewParameters(
            show_demo=False
        ),
    )
    return comparison
