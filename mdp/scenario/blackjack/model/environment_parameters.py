from __future__ import annotations
from dataclasses import dataclass
# import copy

from mdp import common


@dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    pass


# default: EnvironmentParameters = EnvironmentParameters(
#     environment_type=common.EnvironmentType.BLACKJACK,
#     verbose=False,
# )
#
#
# def default_factory() -> EnvironmentParameters:
#     return copy.deepcopy(default)
