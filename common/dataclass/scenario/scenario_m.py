from __future__ import annotations
import dataclasses

from common import enums
from common.dataclass import settings, algorithm_parameters_, policy_parameters_,\
    environment_parameters_, graph_parameters_


@dataclasses.dataclass
class Scenario:
    # mandatory
    environment_type: enums.EnvironmentType
    comparison_type: enums.ComparisonType
    scenario_settings: settings.Settings
    settings_list: list[settings.Settings] = dataclasses.field(default_factory=list)

    # environment
    environment_parameters: environment_parameters_.EnvironmentParameters = \
        dataclasses.field(default_factory=environment_parameters_.default_factory)

    # output
    graph_parameters: graph_parameters_.GraphParameters = \
        dataclasses.field(default_factory=graph_parameters_.default_factory)
    # graph_parameters: dict[str, any] = dataclasses.field(default_factory=dict)  # should be a dataclass

    def __post_init__(self):
        assert self.settings_list
        # Push scenario values or default values into most settings attributes if currently =None
        for settings_ in self.settings_list:
            self._coalesce_settings(settings_)
            self._coalesce_algorithm_parameters(settings_.algorithm_parameters)
            self._coalesce_policy_parameters(settings_.policy_parameters)

    def _coalesce_settings(self, settings_s: settings.Settings):
        """combine settings values
        order of precedence: settings > scenario > default"""
        scenario_s = self.scenario_settings
        default_s = settings.default_settings

        for attribute_name in settings.precedence_attribute_names:
            settings_value = getattr(settings_s, attribute_name)
            scenario_value = getattr(scenario_s, attribute_name)
            default_value = getattr(default_s, attribute_name)
            # order of precedence: settings_ > scenario_ > default
            if settings_value is None:
                if scenario_value is None:
                    setattr(scenario_s, attribute_name, default_value)
                    setattr(settings_s, attribute_name, default_value)
                else:
                    setattr(settings_s, attribute_name, scenario_value)

    def _coalesce_algorithm_parameters(self, settings_ap: algorithm_parameters_.AlgorithmParameters):
        """combine algorithm_parameters
        order of precedence: settings > scenario > default"""
        scenario_ap: algorithm_parameters_.AlgorithmParameters = self.scenario_settings.algorithm_parameters
        default_ap: algorithm_parameters_.AlgorithmParameters = settings.default_settings.algorithm_parameters

        for attribute_name in algorithm_parameters_.precedence_attribute_names:
            settings_value = getattr(settings_ap, attribute_name)
            scenario_value = getattr(scenario_ap, attribute_name)
            default_value = getattr(default_ap, attribute_name)
            # order of precedence: settings_ > scenario_ > default
            if settings_value is None:
                if scenario_value is None:
                    setattr(scenario_ap, attribute_name, default_value)
                    setattr(settings_ap, attribute_name, default_value)
                else:
                    setattr(settings_ap, attribute_name, scenario_value)

    def _coalesce_policy_parameters(self, settings_pp: policy_parameters_.PolicyParameters):
        """combine algorithm_parameters
        order of precedence: settings > scenario > default"""
        scenario_pp: policy_parameters_.PolicyParameters = self.scenario_settings.policy_parameters
        default_pp: policy_parameters_.PolicyParameters = settings.default_settings.policy_parameters

        for attribute_name in policy_parameters_.precedence_attribute_names:
            settings_value = getattr(settings_pp, attribute_name)
            scenario_value = getattr(scenario_pp, attribute_name)
            default_value = getattr(default_pp, attribute_name)
            # order of precedence: settings_ > scenario_ > default
            if settings_value is None:
                if scenario_value is None:
                    setattr(scenario_pp, attribute_name, default_value)
                    setattr(settings_pp, attribute_name, default_value)
                else:
                    setattr(settings_pp, attribute_name, scenario_value)


# https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python-taking-union-o
# order of precedence: settings_ > scenario_ > default
# algorithm_parameters_kwargs = {**default_ap.__dict__, **scenario_ap.__dict__, **settings_ap.__dict__}
# settings_.algorithm_parameters = algorithm_parameters_.AlgorithmParameters(**algorithm_parameters_kwargs)
