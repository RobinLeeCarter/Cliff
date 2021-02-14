from __future__ import annotations
import dataclasses

from common import enums
from common.dataclass import settings
from common.dataclass.scenario import graph_parameters_m


@dataclasses.dataclass
class Scenario:
    # mandatory
    environment_type: enums.EnvironmentType
    comparison_type: enums.ComparisonType
    scenario_settings: settings.Settings
    settings_list: list[settings.Settings] = dataclasses.field(default_factory=list)

    # environment
    environment_parameters: dict[str, any] = dataclasses.field(default_factory=dict)

    # output
    graph_parameters: graph_parameters_m.GraphParameters = \
        dataclasses.field(default_factory=graph_parameters_m.default_factory)
    # graph_parameters: dict[str, any] = dataclasses.field(default_factory=dict)  # should be a dataclass

    def __post_init__(self):
        assert self.settings_list
        # Push scenario values or default values into most settings attributes if currently =None
        for settings_ in self.settings_list:
            self._fully_populate_settings(settings_)

    def _fully_populate_settings(self, settings_: settings.Settings):
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
            scenario_value = getattr(self.scenario_settings, attribute)
            default_value = getattr(settings.default_settings, attribute)
            # order of precedence: settings_ > scenario_ > default
            if settings_value is None:
                if scenario_value is None:
                    setattr(self.scenario_settings, attribute, default_value)
                    setattr(settings_, attribute, default_value)
                else:
                    setattr(settings_, attribute, scenario_value)

        # combine algorithm_parameters
        settings_p = settings_.algorithm_parameters
        scenario_p = self.scenario_settings.algorithm_parameters
        default_p = settings.default_settings.algorithm_parameters
        # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python-taking-union-o
        # order of precedence: settings_ > scenario_ > default
        settings_.algorithm_parameters = {**default_p, **scenario_p, **settings_p}
