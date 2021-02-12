from __future__ import annotations
# from typing import TYPE_CHECKING

from common import enums
from common.dataclass import scenario, settings

# _e = enums.EnvironmentType
# _c = enums.ComparisonType
# _a = enums.AlgorithmType


class Scenarios:
    def __init__(self):
        self.cliff_timestep = self.cliff_timestep_scenario()

    def cliff_timestep_scenario(self) -> scenario.Scenario:
        cliff_timestep = scenario.Scenario(
            environment_type=enums.EnvironmentType.Cliff,
            comparison_type=enums.ComparisonType.EPISODE_BY_TIMESTEP,
            settings_list=[
                settings.Settings(enums.AlgorithmType.Sarsa, {"alpha": 0.5})
            ],
            environment_parameters={"random_wind": False},
            training_episodes=170,
            runs=50
        )
        cliff_timestep.graph_parameters = {
            "y_min": 0,
            "y_max": cliff_timestep.training_episodes
        }
        return cliff_timestep

