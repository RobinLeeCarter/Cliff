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
        runs=100,
        training_episodes=500
    ),
    alpha_min=0.1,
    alpha_max=1.0,
    alpha_step=0.1,
    algorithm_type_list=[
        enums.AlgorithmType.EXPECTED_SARSA,
        enums.AlgorithmType.VQ,
        enums.AlgorithmType.Q_LEARNED,
        enums.AlgorithmType.SARSA
    ]
)
