from __future__ import annotations
from typing import TypeVar, Generic
from abc import ABC, abstractmethod


T = TypeVar('T')


class AbstractDistribution(Generic[T], ABC):

    @abstractmethod
    def draw_one(self) -> T:
        pass
