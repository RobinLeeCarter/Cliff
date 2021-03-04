from __future__ import annotations
# from typing import TYPE_CHECKING
from typing import Optional
import dataclasses
import copy

import utils
from mdp.common import enums
from mdp.common.dataclass import settings, algorithm_parameters
from mdp.common.dataclass.breakdown_parameters import breakdown_parameters_


@dataclasses.dataclass
class BreakdownAlgorithmByAlpha(breakdown_parameters_.BreakdownParameters):
    # derived dataclasses effectively must have default values:
    # https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
    # can supply min, max and step or just supply a list
    alpha_min: Optional[float] = None
    alpha_max: Optional[float] = None
    alpha_step: Optional[float] = None
    alpha_list: list[float] = dataclasses.field(default_factory=list)

    algorithm_type_list: list[enums.AlgorithmType] = dataclasses.field(default_factory=list)

    settings_list: list[settings.Settings] = dataclasses.field(default_factory=list, init=False)

    def __post_init__(self):
        if not self.alpha_list:
            self.alpha_list = utils.float_range(start=self.alpha_min, stop=self.alpha_max, step_size=self.alpha_step)

        for alpha in self.alpha_list:
            for algorithm_type in self.algorithm_type_list:
                settings_ = settings.Settings(
                    algorithm_parameters=algorithm_parameters.AlgorithmParameters(
                        algorithm_type=algorithm_type,
                        alpha=alpha
                    )
                )
                self.settings_list.append(settings_)


default: BreakdownAlgorithmByAlpha = BreakdownAlgorithmByAlpha(
    verbose=breakdown_parameters_.default.verbose,
    alpha_min=0.1,
    alpha_max=1.0,
    alpha_step=0.1
)
