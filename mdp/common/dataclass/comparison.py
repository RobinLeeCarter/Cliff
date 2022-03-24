from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC
from typing import Optional

from mdp.common.dataclass.graph2d_values import Graph2DValues
from mdp.common.dataclass.graph3d_values import Graph3DValues
from mdp.common.dataclass.grid_view_parameters import GridViewParameters

from mdp.common.dataclass.environment_parameters import EnvironmentParameters
from mdp.common.dataclass.settings import Settings
from mdp.common.dataclass.breakdown_parameters.breakdown_parameters import BreakdownParameters
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
    graph2d_values: Optional[Graph2DValues] = None

    # 3D graph output
    graph3d_values: Optional[Graph3DValues] = None

    # grid view output
    grid_view_parameters: GridViewParameters = field(default_factory=GridViewParameters)

    def __post_init__(self):
        if isinstance(self.breakdown_parameters, BreakdownAlgorithmByAlpha):
            self.settings_list = self.breakdown_parameters.build_settings_list(self.comparison_settings)

        assert self.settings_list
