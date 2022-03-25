from __future__ import annotations
from typing import TypeVar, Generic
from abc import ABC

from mdp.model.base.base_model import BaseModel
from mdp.model.non_tabular.algorithm.non_tabular_algorithm import NonTabularAlgorithm
from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent
from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)
Environment = TypeVar('Environment', bound=NonTabularEnvironment)


class NonTabularModel(Generic[State, Action, Environment],
                      BaseModel[Environment, NonTabularAgent[State, Action]],
                      ABC):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

    def _create_agent(self) -> NonTabularAgent[State, Action]:
        return NonTabularAgent[State, Action](self.environment)

    @property
    def algorithm(self) -> NonTabularAlgorithm:
        algorithm = self.trainer.algorithm
        assert isinstance(algorithm, NonTabularAlgorithm)
        return algorithm
