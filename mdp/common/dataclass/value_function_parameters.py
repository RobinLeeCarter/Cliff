from __future__ import annotations
from dataclasses import dataclass
# from typing import Optional

from mdp.common.enums import ValueFunctionType


@dataclass
class ValueFunctionParameters:
    value_function_type: ValueFunctionType
    initial_value: float = 0.0
    requires_delta_w: bool = False
