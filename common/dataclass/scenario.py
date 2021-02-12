from __future__ import annotations
import dataclasses
from typing import Optional

from common import constants, enums
from common.dataclass import settings


@dataclasses.dataclass
class Scenario:
    environment_type: enums.EnvironmentType
    comparison_type: enums.ComparisonType

    gamma: float = constants.GAMMA
    environment_kwargs: dict[str, any] = dataclasses.field(default_factory=dict)

    scenario_settings: Optional[settings.Settings] = None
    settings_list: list[settings.Settings] = dataclasses.field(default_factory=list)

    moving_average_window_size: int = constants.MOVING_AVERAGE_WINDOW_SIZE
    graph_parameters: dict[str, any] = dataclasses.field(default_factory=dict)
