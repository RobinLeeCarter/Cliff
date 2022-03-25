from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
from abc import ABC

from mdp import common
from mdp.model.tabular.environment.tabular_environment_parameters import TabularEnvironmentParameters


@dataclass
class PositionMoveEnvironmentParameters(TabularEnvironmentParameters, ABC):
    actions_list: Optional[common.ActionsList] = None
