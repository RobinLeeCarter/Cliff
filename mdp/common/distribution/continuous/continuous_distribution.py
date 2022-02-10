from __future__ import annotations
from typing import TypeVar, Generic
from abc import ABC

from mdp.common.distribution.distribution import Distribution

T = TypeVar('T')


class ContinuousDistribution(Generic[T], Distribution[T], ABC):
    pass
