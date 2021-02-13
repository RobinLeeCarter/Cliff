from __future__ import annotations
# from typing import TYPE_CHECKING
import dataclasses
from typing import Optional

import utils
from common import enums
from common.dataclass import settings
from common.dataclass.scenario import scenario_m


@dataclasses.dataclass
class AlgorithmByAlpha(scenario_m.Scenario):
    # derived dataclasses effectively must have default values:
    # https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
    alpha_min: Optional[float] = None
    alpha_max: Optional[float] = None
    alpha_step: Optional[float] = None
    algorithm_type_list: list[enums.AlgorithmType] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        assert self.alpha_min is not None
        assert self.alpha_max is not None
        assert self.alpha_step is not None
        assert self.algorithm_type_list

        self._alpha_list = utils.float_range(start=self.alpha_min, stop=self.alpha_max, step_size=self.alpha_step)
        for alpha in self._alpha_list:
            for algorithm_type in self.algorithm_type_list:
                settings_ = settings.Settings(
                    algorithm_type=algorithm_type,
                    algorithm_parameters={"alpha": alpha}
                )
                self.settings_list.append(settings_)

        super().__post_init__()
