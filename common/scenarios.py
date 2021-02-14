from __future__ import annotations
import copy

from common import enums, dataclass


# do we want random_wind to be a parameter?
windy_timestep = dataclass.Scenario(
    environment_type=enums.EnvironmentType.WINDY,
    comparison_type=enums.ComparisonType.EPISODE_BY_TIMESTEP,
    scenario_settings=dataclass.Settings(
        runs=50,
        training_episodes=170,
        review_every_step=True
        # algorithm_parameters=dataclass.AlgorithmParameters(
        #     initial_q_value=5.0,
        #     initial_v_value=6.0
        # )
    ),
    settings_list=[
        dataclass.Settings(
            algorithm_type=enums.AlgorithmType.SARSA,
            algorithm_parameters=dataclass.AlgorithmParameters(
                alpha=0.5,
                initial_q_value=-20.0
            )
        )
    ]
)

random_windy_timestep = copy.deepcopy(windy_timestep)
random_windy_timestep.environment_parameters.random_wind = True

cliff_alpha = dataclass.AlgorithmByAlpha(
    environment_type=enums.EnvironmentType.CLIFF,
    comparison_type=enums.ComparisonType.RETURN_BY_ALPHA,
    scenario_settings=dataclass.Settings(
        runs=10,
        training_episodes=100
    ),
    alpha_min=0.1,
    alpha_max=1.0,
    alpha_step=0.1,
    algorithm_type_list=[
        enums.AlgorithmType.EXPECTED_SARSA,
        enums.AlgorithmType.VQ,
        enums.AlgorithmType.Q_LEARNING,
        enums.AlgorithmType.SARSA
    ],
    graph_parameters=dataclass.GraphParameters(
        y_min=-140,
        y_max=0
    )
)

cliff_episode = dataclass.Scenario(
    environment_type=enums.EnvironmentType.CLIFF,
    comparison_type=enums.ComparisonType.RETURN_BY_EPISODE,
    scenario_settings=dataclass.Settings(
        runs=10,
        training_episodes=500
    ),
    settings_list=[
        dataclass.Settings(algorithm_type=enums.AlgorithmType.EXPECTED_SARSA,
                           algorithm_parameters=dataclass.AlgorithmParameters(alpha=0.9)),
        dataclass.Settings(algorithm_type=enums.AlgorithmType.VQ,
                           algorithm_parameters=dataclass.AlgorithmParameters(alpha=0.2)),
        dataclass.Settings(algorithm_type=enums.AlgorithmType.Q_LEARNING,
                           algorithm_parameters=dataclass.AlgorithmParameters(alpha=0.5)),
        dataclass.Settings(algorithm_type=enums.AlgorithmType.SARSA,
                           algorithm_parameters=dataclass.AlgorithmParameters(alpha=0.5)),
    ],
    graph_parameters=dataclass.GraphParameters(
        moving_average_window_size=19,
        y_min=-100,
        y_max=0
    )
)
