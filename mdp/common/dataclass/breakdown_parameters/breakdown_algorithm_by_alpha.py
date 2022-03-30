from __future__ import annotations
import copy
from dataclasses import dataclass, field

import utils
from mdp.common.enums import AlgorithmType
from mdp.common.dataclass.settings import Settings
from mdp.common.dataclass.breakdown_parameters.breakdown_parameters import BreakdownParameters


@dataclass
class BreakdownAlgorithmByAlpha(BreakdownParameters):
    """can supply min, max and step or just supply a list"""
    alpha_min: float = 0.1
    alpha_max: float = 1.0
    alpha_step: float = 0.1
    alpha_list: list[float] = field(default_factory=list)
    algorithm_type_list: list[AlgorithmType] = field(default_factory=list)

    def __post_init__(self):
        if not self.alpha_list:
            self.alpha_list = utils.float_range_step_size(start=self.alpha_min, stop=self.alpha_max, step_size=self.alpha_step)

    def build_settings_list(self, settings_template: Settings) -> list[Settings]:
        settings_list: list[Settings] = []
        for alpha in self.alpha_list:
            for algorithm_type in self.algorithm_type_list:
                settings = copy.deepcopy(settings_template)
                settings.algorithm_parameters.algorithm_type = algorithm_type
                settings.algorithm_parameters.alpha = alpha
                settings_list.append(settings)
        return settings_list
