from __future__ import annotations
# from typing import TYPE_CHECKING
import dataclasses
from typing import Optional

import utils
from common import enums
from common.dataclass import settings, algorithm_parameters
from common.dataclass.comparison_parameters import comparison_parameters_


@dataclasses.dataclass
class ComparisonAlgorithmByAlpha(comparison_parameters_.ComparisonParameters):
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
        # assert self.alpha_min is not None
        # assert self.alpha_max is not None
        # assert self.alpha_step is not None
        assert self.algorithm_type_list

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
        assert self.settings_list
