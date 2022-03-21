from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Callable, TYPE_CHECKING    # , TypeVar

if TYPE_CHECKING:
    from mdp.model.base.environment.base_environment import BaseEnvironment

# from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
# from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction
#
# State = TypeVar('State', bound=NonTabularState)
# Action = TypeVar('Action', bound=NonTabularAction)


class BaseEpisode(ABC):
    """Just makes a record laid out in the standard way with Reward, State, Action for each _t"""
    def __init__(self,
                 environment: BaseEnvironment,
                 gamma: float,
                 step_callback: Optional[Callable[[], bool]] = None,
                 record_first_visits: bool = False):
        self._environment: BaseEnvironment = environment
        self.gamma: float = gamma
        self._step_callback: Optional[Callable[[], bool]] = step_callback
        self.record_first_visits = record_first_visits

    @property
    @abstractmethod
    def max_t(self) -> int:
        ...

    @property
    @abstractmethod
    def total_return(self) -> float:
        ...
