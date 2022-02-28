from __future__ import annotations
from typing import TypeVar, Generic
from abc import ABC, abstractmethod


T_co = TypeVar('T_co', covariant=True)


class Distribution(Generic[T_co], ABC):
    @abstractmethod
    def draw_one(self) -> T_co:
        pass
