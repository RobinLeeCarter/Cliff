from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from mdp.model.environment.action import Action


@dataclass(frozen=True)
class NonTabularAction(Action, ABC):
    __discrete_tuple: tuple = field(init=False, hash=False, compare=False)

    def __post_init__(self):
        # from: https://stackoverflow.com/q/53756788
        object.__setattr__(self, "_NonTabularAction__discrete_tuple", self._get_discrete_tuple())

    @property
    def values(self) -> tuple:
        return self.__discrete_tuple

    @abstractmethod
    def _get_discrete_tuple(self) -> tuple:
        pass
