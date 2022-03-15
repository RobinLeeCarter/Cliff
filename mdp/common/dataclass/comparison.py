from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC

import utils
from mdp.common.dataclass import environment_parameters_, settings
from mdp.common.dataclass import graph_values_, graph3d_values_, grid_view_parameters_
from mdp.common.dataclass.breakdown_parameters import breakdown_parameters_, breakdown_algorithm_by_alpha
from mdp.common.enums import ParallelContextType


@dataclass
class Comparison(ABC):
    """
    The parameters used by the Application during it's build() stage
    """
    environment_parameters: environment_parameters_.EnvironmentParameters

    # training session settings
    comparison_settings: settings.Settings = field(default_factory=settings.default_factory)
    settings_list: list[settings.Settings] = field(default_factory=list)
    settings_list_multiprocessing: ParallelContextType = ParallelContextType.NONE

    # breakdown
    breakdown_parameters: breakdown_parameters_.BreakdownParameters = \
        field(default_factory=breakdown_parameters_.default_factory)

    # 2D graph output
    graph_values: graph_values_.GraphValues = \
        field(default_factory=graph_values_.default_factory)

    # 3D graph output
    graph3d_values: graph3d_values_.Graph3DValues = \
        field(default_factory=graph3d_values_.default_factory)

    # grid view output
    grid_view_parameters: grid_view_parameters_.GridViewParameters = \
        field(default_factory=grid_view_parameters_.default_factory)

    def __post_init__(self):
        # Push comparison values or default values into most settings attributes if currently =None
        # model
        # this is now done in the derived classes using their defaults
        # utils.set_none_to_default(self.environment_parameters, environment_parameters.default)

        if isinstance(self.breakdown_parameters, breakdown_algorithm_by_alpha.BreakdownAlgorithmByAlpha):
            utils.set_none_to_default(self.breakdown_parameters, breakdown_algorithm_by_alpha.default)
            self.settings_list = self.breakdown_parameters.settings_list
        else:
            utils.set_none_to_default(self.breakdown_parameters, breakdown_parameters_.default)

        assert self.settings_list
        self.comparison_settings.set_none_to_default(default_=settings.default)
        for settings_ in self.settings_list:
            settings_.set_none_to_default(default_=self.comparison_settings)

        # view
        utils.set_none_to_default(self.graph_values, graph_values_.default)
        utils.set_none_to_default(self.grid_view_parameters, grid_view_parameters_.default)
