from __future__ import annotations
# from typing import TYPE_CHECKING

from common import enums, constants
from common.dataclass import scenario_m, settings

# _e = enums.EnvironmentType
# _c = enums.ComparisonType
# _a = enums.AlgorithmType


class Scenarios:
    def __init__(self):
        self.windy_timestep = self._build_windy_timestep()

    def _build_windy_timestep(self) -> scenario_m.Scenario:
        scenario_ = scenario_m.Scenario(
            environment_type=enums.EnvironmentType.WINDY,
            environment_kwargs={"random_wind": False},
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
                                          "initial_q_value": 0.0}
                )
            ]
        )

        for settings_ in scenario_.settings_list:
            self._fully_populate_settings(scenario_, settings_)

        # scenario_.graph_parameters = {
        #     "y_min": 0,
        #     "y_max": scenario_.scenario_settings.training_episodes
        # }
        return scenario_

    def _fully_populate_settings(self, scenario_: scenario_m.Scenario, settings_: settings.Settings):
        # order of precedence: settings_ > scenario_ > default
        attributes: list[str] = [
            'gamma',
            'runs',
            'run_print_frequency',
            'training_episodes',
            'episode_length_timeout',
            'episode_print_frequency',
            'episode_to_start_recording',
            'episode_recording_frequency',
            'review_every_step'
        ]
        for attribute in attributes:
            settings_value = getattr(settings_, attribute)
            scenario_value = getattr(scenario_.scenario_settings, attribute)
            default_value = getattr(constants.default_settings, attribute)
            # order of precedence: settings_ > scenario_ > default
            if settings_value is None:
                if scenario_value is None:
                    setattr(settings_, attribute, default_value)
                else:
                    setattr(settings_, attribute, scenario_value)

        settings_p = settings_.algorithm_parameters
        scenario_p = scenario_.scenario_settings.algorithm_parameters
        default_p = constants.default_settings.algorithm_parameters

        # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python-taking-union-o
        # order of precedence: settings_ > scenario_ > default
        settings_.algorithm_parameters = {**default_p, **scenario_p, **settings_p}
