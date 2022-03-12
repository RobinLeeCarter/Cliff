from __future__ import annotations
from typing import TypeVar, Generic
from abc import ABC

from mdp.model.non_tabular.agent.non_tabular_agent import NonTabularAgent

from mdp.model.general.general_model import GeneralModel
from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)
Environment = TypeVar('Environment', bound=NonTabularEnvironment)


class NonTabularModel(Generic[State, Action, Environment],
                      GeneralModel[Environment, NonTabularAgent[State, Action]],
                      ABC):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

    def _create_agent(self) -> NonTabularAgent[State, Action]:
        return NonTabularAgent[State, Action](self.environment)
