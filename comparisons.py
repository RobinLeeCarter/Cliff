from __future__ import annotations
import copy

import common


# do we want random_wind to be a parameter?
windy_timestep = common.Comparison(
    environment_parameters=common.EnvironmentParameters(
        environment_type=common.EnvironmentType.WINDY,
        actions_list=common.ActionsList.FOUR_MOVES,
    ),
    comparison_settings=common.Settings(
        runs=50,
        training_episodes=170,
        review_every_step=True,
    ),
    breakdown_parameters=common.BreakdownParameters(
        breakdown_type=common.BreakdownType.EPISODE_BY_TIMESTEP,
    ),
    settings_list=[
        common.Settings(
            algorithm_parameters=common.AlgorithmParameters(
                algorithm_type=common.AlgorithmType.SARSA,
                alpha=0.5,
                initial_q_value=-20.0,
            )
        )
    ],
)

random_windy_timestep = copy.deepcopy(windy_timestep)
random_windy_timestep.environment_parameters.random_wind = True

cliff_alpha = common.Comparison(
    environment_parameters=common.EnvironmentParameters(
        environment_type=common.EnvironmentType.CLIFF,
        actions_list=common.ActionsList.FOUR_MOVES,
    ),
    comparison_settings=common.Settings(
        runs=10,
        training_episodes=100,
    ),
    breakdown_parameters=common.BreakdownAlgorithmByAlpha(
        breakdown_type=common.BreakdownType.RETURN_BY_ALPHA,
        alpha_min=0.1,
        alpha_max=1.0,
        alpha_step=0.1,
        algorithm_type_list=[
            common.AlgorithmType.EXPECTED_SARSA,
            common.AlgorithmType.VQ,
            common.AlgorithmType.Q_LEARNING,
            common.AlgorithmType.SARSA
        ],
    ),
    graph_parameters=common.GraphParameters(
        y_min=-140,
        y_max=0,
    ),
)

cliff_episode = common.Comparison(
    environment_parameters=common.EnvironmentParameters(
        environment_type=common.EnvironmentType.CLIFF,
        actions_list=common.ActionsList.FOUR_MOVES,
    ),
    comparison_settings=common.Settings(
        runs=10,
        training_episodes=500,
    ),
    breakdown_parameters=common.BreakdownParameters(
        breakdown_type=common.BreakdownType.RETURN_BY_EPISODE
    ),
    settings_list=[
        common.Settings(algorithm_parameters=common.AlgorithmParameters(
            algorithm_type=common.AlgorithmType.EXPECTED_SARSA,
            alpha=0.9
        )),
        common.Settings(algorithm_parameters=common.AlgorithmParameters(
            algorithm_type=common.AlgorithmType.VQ,
            alpha=0.2
        )),
        common.Settings(algorithm_parameters=common.AlgorithmParameters(
            algorithm_type=common.AlgorithmType.Q_LEARNING,
            alpha=0.5
        )),
        common.Settings(algorithm_parameters=common.AlgorithmParameters(
            algorithm_type=common.AlgorithmType.SARSA,
            alpha=0.5
        )),
    ],
    graph_parameters=common.GraphParameters(
        moving_average_window_size=19,
        y_min=-100,
        y_max=0
    ),
)
