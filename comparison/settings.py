from __future__ import annotations
from dataclasses import dataclass
from typing import Type, Dict

import algorithm


@dataclass(frozen=True)
class Settings:
    algorithm_type: Type[algorithm.EpisodicAlgorithm]
    parameters: Dict[str, any]
