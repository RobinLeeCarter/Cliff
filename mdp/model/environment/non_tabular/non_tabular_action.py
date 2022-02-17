from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np

from mdp.model.environment.action import Action


@dataclass(frozen=True)
class NonTabularAction(Action, ABC):
    categories: np.ndarray = field(init=False, repr=False, hash=False, compare=False)

    def __post_init__(self):
        # from: https://stackoverflow.com/q/53756788
        object.__setattr__(self, "categories", np.array(self._get_categories(), dtype=object))

    @abstractmethod
    def _get_categories(self) -> list:
        pass