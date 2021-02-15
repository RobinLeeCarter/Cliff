from __future__ import annotations
# from typing import TYPE_CHECKING
from typing import Optional
import dataclasses
import copy

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

    def apply_default_to_nones(self, default_: ComparisonAlgorithmByAlpha):
        super().apply_default_to_nones(default_)
        attribute_names: list[str] = [
            'alpha_min',
            'alpha_max',
            'alpha_step',
            'alpha_list'
        ]
        for attribute_name in attribute_names:
            attribute = self.__getattribute__(attribute_name)
            if attribute is None:
                default_value = default_.__getattribute__(attribute_name)
                self.__setattr__(attribute_name, default_value)


default: ComparisonAlgorithmByAlpha = ComparisonAlgorithmByAlpha(
    verbose=comparison_parameters_.default.verbose,
    alpha_min=0.1,
    alpha_max=1.0,
    alpha_step=0.1
)


def default_factory() -> ComparisonAlgorithmByAlpha:
    return copy.deepcopy(default)
