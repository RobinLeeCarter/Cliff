from __future__ import annotations
from abc import ABC, abstractmethod

from mdp import common


class BaseComparisonBuilder(ABC):
    def __init__(self, **_ignored):
        pass

    @abstractmethod
    def create(self) -> common.Comparison:
        ...
