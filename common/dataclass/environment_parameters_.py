from __future__ import annotations
import dataclasses
# from typing import Optional


@dataclasses.dataclass
class EnvironmentParameters:
    random_wind: bool = False
    verbose: bool = False


def default_factory() -> EnvironmentParameters:
    return EnvironmentParameters()
