from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Type, Optional

from mdp import common


class BaseComparisonBuilder(ABC):
    type_registry: dict[common.ComparisonType, Type[BaseComparisonBuilder]] = {}

    def __init_subclass__(cls, comparison_type: Optional[common.ComparisonType] = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if comparison_type:
            BaseComparisonBuilder.type_registry[comparison_type] = cls

    @abstractmethod
    def create(self) -> common.Comparison:
        ...
