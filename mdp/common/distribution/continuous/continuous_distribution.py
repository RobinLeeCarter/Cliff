from __future__ import annotations
from typing import TypeVar
from abc import ABC

from mdp.common.distribution.distribution import Distribution

T_co = TypeVar('T_co', covariant=True)


class ContinuousDistribution(Distribution[T_co], ABC):
    pass
