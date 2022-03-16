from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC

import utils
from mdp.common.dataclass.graph2d_values_ import Graph2DValues
from mdp.common.dataclass.graph3d_values_ import Graph3DValues
from mdp.common.dataclass.grid_view_parameters_ import GridViewParameters

from mdp.common.dataclass.environment_parameters_ import EnvironmentParameters
from mdp.common.dataclass.settings import Settings
from mdp.common.dataclass.breakdown_parameters.breakdown_parameters_ import BreakdownParameters
from mdp.common.dataclass.breakdown_parameters.breakdown_algorithm_by_alpha import BreakdownAlgorithmByAlpha
from mdp.common.enums import ParallelContextType


@dataclass
class Comparison(ABC):
    """
    The parameters used by the Application during it's build() stage
    """
    environment_parameters: EnvironmentParameters

    # training session settings
    comparison_settings: Settings = field(default_factory=Settings)
    settings_list: list[Settings] = field(default_factory=list)
    settings_list_multiprocessing: ParallelContextType = ParallelContextType.NONE

    # breakdown
    breakdown_parameters: BreakdownParameters = field(default_factory=BreakdownParameters)

    # 2D graph output
    graph2d_values: Graph2DValues = field(default_factory=Graph2DValues)

    # 3D graph output
    graph3d_values: Graph3DValues = field(default_factory=Graph3DValues)

    # grid view output
    grid_view_parameters: GridViewParameters = field(default_factory=GridViewParameters)

    def __post_init__(self):
        # Push comparison values or default values into most settings attributes if currently =None
        # model
        # this is now done in the derived classes using their defaults
        # utils.set_none_to_default(self.environment_parameters, environment_parameters.default)

        if isinstance(self.breakdown_parameters, BreakdownAlgorithmByAlpha):
            self.settings_list = self.breakdown_parameters.build_settings_list(self.comparison_settings)

        assert self.settings_list
        # self.comparison_settings.set_none_to_default(default_=settings.default)
        # for settings_ in self.settings_list:
        #     settings_.set_none_to_default(default_=self.comparison_settings)

        # view
        # utils.set_none_to_default(self.graph_values, graph_values_.default)
        # utils.set_none_to_default(self.grid_view_parameters, grid_view_parameters_.default)
