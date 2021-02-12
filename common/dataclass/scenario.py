from __future__ import annotations
import dataclasses

from common import constants, enums
from common.dataclass import settings


@dataclasses.dataclass
class Scenario:
    # mandatory
    environment_type: enums.EnvironmentType
    comparison_type: enums.ComparisonType
    scenario_settings: settings.Settings
    settings_list: list[settings.Settings]

    # environment
    gamma: float = constants.GAMMA
    environment_kwargs: dict[str, any] = dataclasses.field(default_factory=dict)

    # output
    graph_parameters: dict[str, any] = dataclasses.field(default_factory=dict)
