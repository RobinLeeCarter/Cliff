from __future__ import annotations

from mdp import common


def cliff_episode() -> common.Comparison:
    comparison = common.Comparison(
        environment_parameters=common.EnvironmentParameters(
            environment_type=common.EnvironmentType.RACETRACK,
        ),
        comparison_settings=common.Settings(
            runs=1,
            training_episodes=10_000,
            # display_every_step=True,
        ),
        breakdown_parameters=common.BreakdownParameters(
            breakdown_type=common.BreakdownType.RETURN_BY_EPISODE
        ),
        settings_list=[
            # common.Settings(algorithm_parameters=common.AlgorithmParameters(
            #     algorithm_type=common.AlgorithmType.EXPECTED_SARSA,
            #     alpha=0.9
            # )),
            # common.Settings(algorithm_parameters=common.AlgorithmParameters(
            #     algorithm_type=common.AlgorithmType.VQ,
            #     alpha=0.2
            # )),
            # common.Settings(algorithm_parameters=common.AlgorithmParameters(
            #     algorithm_type=common.AlgorithmType.Q_LEARNING,
            #     alpha=0.5
            # )),
            common.Settings(algorithm_parameters=common.AlgorithmParameters(
                algorithm_type=common.AlgorithmType.OFF_POLICY_MC_CONTROL
            )),
        ],
        graph_values=common.GraphValues(
            moving_average_window_size=19,
            y_min=-100,
            y_max=0
        ),
    )
    return comparison
