from __future__ import annotations
import dataclasses

from mdp.common.dataclass import settings, graph_values, grid_view_parameters, environment_parameters
from mdp.common.dataclass.breakdown_parameters import breakdown_parameters_, breakdown_algorithm_by_alpha


@dataclasses.dataclass
class Comparison:
    # environment
    environment_parameters: environment_parameters.EnvironmentParameters = \
        dataclasses.field(default_factory=environment_parameters.default_factory)

    # training session settings
    comparison_settings: settings.Settings = dataclasses.field(default_factory=settings.default_factory)
    settings_list: list[settings.Settings] = dataclasses.field(default_factory=list)

    # breakdown
    breakdown_parameters: breakdown_parameters_.BreakdownParameters = \
        dataclasses.field(default_factory=breakdown_parameters_.default_factory)

    # graph output
    graph_values: graph_values.GraphValues = \
        dataclasses.field(default_factory=graph_values.default_factory)

    # grid view output
    grid_view_parameters: grid_view_parameters.GridViewParameters = \
        dataclasses.field(default_factory=grid_view_parameters.default_factory)

    def __post_init__(self):
        # Push comparison values or default values into most settings attributes if currently =None
        self.environment_parameters.apply_default_to_nones(default_=environment_parameters.default)

        if isinstance(self.breakdown_parameters, breakdown_algorithm_by_alpha.BreakdownAlgorithmByAlpha):
            self.breakdown_parameters.apply_default_to_nones(default_=breakdown_algorithm_by_alpha.default)
            self.settings_list = self.breakdown_parameters.settings_list
        else:
            self.breakdown_parameters.apply_default_to_nones(default_=breakdown_parameters_.default)

        assert self.settings_list
        self.comparison_settings.apply_default_to_nones(default_=settings.default)
        for settings_ in self.settings_list:
            settings_.apply_default_to_nones(default_=self.comparison_settings)

        self.graph_values.apply_default_to_nones(default_=graph_values.default)
        self.grid_view_parameters.apply_default_to_nones(default_=grid_view_parameters.default)
