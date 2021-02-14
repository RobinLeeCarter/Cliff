from __future__ import annotations
import copy

import common


# do we want random_wind to be a parameter?
windy_timestep = common.Scenario(
    environment_parameters=common.EnvironmentParameters(
        environment_type=common.EnvironmentType.WINDY,
        actions_list=common.ActionsList.FOUR_MOVES
    ),
    scenario_settings=common.Settings(
        runs=50,
        training_episodes=170,
        review_every_step=True,
        # algorithm_parameters=common.AlgorithmParameters(
        #     initial_q_value=5.0,
        #     initial_v_value=6.0
        # )
    ),
    comparison_parameters=common.ComparisonParameters(
        comparison_type=common.ComparisonType.EPISODE_BY_TIMESTEP
    ),
    settings_list=[
        common.Settings(
            algorithm_parameters=common.AlgorithmParameters(
                algorithm_type=common.AlgorithmType.SARSA,
                alpha=0.5,
                initial_q_value=-20.0
            )
        )
    ]
)

random_windy_timestep = copy.deepcopy(windy_timestep)
random_windy_timestep.environment_parameters.random_wind = True

cliff_alpha = common.Scenario(
    environment_parameters=common.EnvironmentParameters(
        environment_type=common.EnvironmentType.CLIFF,
        actions_list=common.ActionsList.FOUR_MOVES
    ),
    scenario_settings=common.Settings(
        runs=10,
        training_episodes=100
    ),
    comparison_parameters=common.ComparisonAlgorithmByAlpha(
        comparison_type=common.ComparisonType.RETURN_BY_ALPHA,
        alpha_min=0.1,
        alpha_max=1.0,
        alpha_step=0.1,
        algorithm_type_list=[
            common.AlgorithmType.EXPECTED_SARSA,
            common.AlgorithmType.VQ,
            common.AlgorithmType.Q_LEARNING,
            common.AlgorithmType.SARSA
        ]
    ),
    graph_parameters=common.GraphParameters(
        y_min=-140,
        y_max=0
    )
)

cliff_episode = common.Scenario(
    environment_parameters=common.EnvironmentParameters(
        environment_type=common.EnvironmentType.CLIFF,
        actions_list=common.ActionsList.FOUR_MOVES
    ),
    scenario_settings=common.Settings(
        runs=10,
        training_episodes=500
        # policy_parameters=common.PolicyParameters(epsilon=0.2)
    ),
    comparison_parameters=common.ComparisonParameters(
        comparison_type=common.ComparisonType.RETURN_BY_EPISODE
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
    )
)
