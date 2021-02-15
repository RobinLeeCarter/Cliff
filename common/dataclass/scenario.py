from __future__ import annotations
import dataclasses

from common.dataclass import settings, environment_parameters, graph_parameters
from common.dataclass.comparison_parameters import comparison_parameters_, algorithm_by_alpha


@dataclasses.dataclass
class Scenario:
    # environment
    environment_parameters: environment_parameters.EnvironmentParameters = \
        dataclasses.field(default_factory=environment_parameters.default_factory)

    # training session settings
    scenario_settings: settings.Settings = dataclasses.field(default_factory=settings.default_factory)
    settings_list: list[settings.Settings] = dataclasses.field(default_factory=list)

    # comparison
    comparison_parameters: comparison_parameters_.ComparisonParameters = \
        dataclasses.field(default_factory=comparison_parameters_.default_factory)

    # graph output
    graph_parameters: graph_parameters.GraphParameters = \
        dataclasses.field(default_factory=graph_parameters.default_factory)

    def __post_init__(self):
        # Push scenario values or default values into most settings attributes if currently =None
        self.environment_parameters.apply_default_to_nones(default_=environment_parameters.default)

        if isinstance(self.comparison_parameters, algorithm_by_alpha.ComparisonAlgorithmByAlpha):
            self.comparison_parameters.apply_default_to_nones(default_=algorithm_by_alpha.default)
            self.settings_list = self.comparison_parameters.settings_list
        else:
            self.comparison_parameters.apply_default_to_nones(default_=comparison_parameters_.default)

        assert self.settings_list
        self.scenario_settings.apply_default_to_nones(default_=settings.default)
        for settings_ in self.settings_list:
            settings_.apply_default_to_nones(default_=self.scenario_settings)

        self.graph_parameters.apply_default_to_nones(default_=graph_parameters.default)
