from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from mdp import common


class GeneralComparisonBuilder(ABC):
    def __init__(self, **_ignored):
        self._comparison: Optional[common.Comparison] = None

    @abstractmethod
    def create(self) -> common.Comparison:
        ...
