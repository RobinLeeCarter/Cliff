from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.non_tabular.non_tabular_environment import NonTabularEnvironment
    from mdp.model.environment.non_tabular.non_tabular_state import NonTabularState
    from mdp.model.feature.feature import Feature


# Note: For now StateFunctions will assume the weights are a vector, there is feature and environment is unknown
# Could refactor in the future to add a level and broaden the definition compound features
# and to memory-based state functions
class StateFunction(ABC):
    def __init__(self,
                 environment_: NonTabularEnvironment,
                 feature: Feature,
                 initial_value: float
                 ):
        """
        :param environment_: environment state function is working in
        :param feature: fully formed feature (need max_size property to size the weight vector)
        :param initial_value: initial value for the weights (else zero)
        """
        # TODO: do we need environment?
        self._environment: NonTabularEnvironment = environment_
        self._feature: Feature = feature
        self._initial_value: float = initial_value

        self._size: int = self._feature.max_size
        self._w: np.ndarray = np.full(shape=self._size, fill_value=initial_value, dtype=float)

    @abstractmethod
    def __getitem__(self, state: NonTabularState) -> float:
        pass
