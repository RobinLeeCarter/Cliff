from __future__ import annotations
import copy

from common import enums
from common.dataclass import scenario, settings


# do we want random_wind to be a parameter?
windy_timestep = scenario.Scenario(
    environment_type=enums.EnvironmentType.WINDY,
    environment_parameters={"random_wind": False},
    comparison_type=enums.ComparisonType.EPISODE_BY_TIMESTEP,
    scenario_settings=settings.Settings(
        runs=50,
        training_episodes=170,
        review_every_step=True
    ),
    settings_list=[
        settings.Settings(
            algorithm_type=enums.AlgorithmType.SARSA,
            algorithm_parameters={"alpha": 0.5,
                                  "initial_q_value": -20.0}
        )
    ]
)

random_windy_timestep = copy.deepcopy(windy_timestep)
random_windy_timestep.environment_parameters["random_wind"] = True

cliff_alpha = scenario.AlgorithmByAlpha(
    environment_type=enums.EnvironmentType.CLIFF,
    comparison_type=enums.ComparisonType.RETURN_BY_ALPHA,
    scenario_settings=settings.Settings(
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
    graph_parameters={"y_min": -140, "y_max:": 0}   # should be a dataclass
)

cliff_episode = scenario.Scenario(
    environment_type=enums.EnvironmentType.CLIFF,
    comparison_type=enums.ComparisonType.RETURN_BY_EPISODE,
    scenario_settings=settings.Settings(
        runs=10,
        training_episodes=500
    ),
    settings_list=[
        settings.Settings(algorithm_type=enums.AlgorithmType.EXPECTED_SARSA,
                          algorithm_parameters={"alpha": 0.9}),
        settings.Settings(algorithm_type=enums.AlgorithmType.VQ,
                          algorithm_parameters={"alpha": 0.2}),
        settings.Settings(algorithm_type=enums.AlgorithmType.Q_LEARNING,
                          algorithm_parameters={"alpha": 0.5}),
        settings.Settings(algorithm_type=enums.AlgorithmType.SARSA,
                          algorithm_parameters={"alpha": 0.5})
    ],
    graph_parameters={
        "moving_average_window_size": 19,
        "y_min": -100,
        "y_max": 0
    }
)
