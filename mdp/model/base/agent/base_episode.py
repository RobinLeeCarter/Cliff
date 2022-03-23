from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.base.environment.base_environment import BaseEnvironment


class BaseEpisode(ABC):
    """Just makes a record laid out in the standard way with Reward, State, Action for each _t"""
    def __init__(self,
                 environment: BaseEnvironment,
                 gamma: float,
                 step_callback: Optional[Callable[[], bool]] = None):
        self._environment: BaseEnvironment = environment
        self.gamma: float = gamma
        self._step_callback: Optional[Callable[[], bool]] = step_callback

    @property
    @abstractmethod
    def max_t(self) -> int:
        ...

    @property
    @abstractmethod
    def total_return(self) -> float:
        ...
