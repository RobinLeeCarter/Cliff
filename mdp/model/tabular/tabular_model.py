from __future__ import annotations
from typing import Optional, TYPE_CHECKING, TypeVar
from abc import abstractmethod

if TYPE_CHECKING:
    from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
    # from mdp.model.tabular.agent.episode import Episode
    from mdp.model.breakdown.breakdown import Breakdown

from mdp import common
# from mdp.scenarios.factory import environment_factory
from mdp.model.tabular.agent.agent import Agent
from mdp.model.breakdown import breakdown_factory
from mdp.model.trainer.trainer import Trainer
from mdp.model.trainer.parallel_trainer import ParallelTrainer

from mdp.model.general.model import Model
from mdp.model.tabular.environment.tabular_state import TabularState
from mdp.model.tabular.environment.tabular_action import TabularAction

State = TypeVar('State', bound=TabularState)
Action = TypeVar('Action', bound=TabularAction)


class TabularModel(Model):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.environment: Optional[TabularEnvironment] = None
        self.agent: Optional[Agent] = None
        # self.breakdown: Optional[Breakdown] = None
        # self.trainer: Optional[Trainer] = None
        # self.parallel_trainer: Optional[ParallelTrainer] = None

    @abstractmethod
    def _create_environment(self, environment_parameters: common.EnvironmentParameters):
        pass

    def _create_agent(self):
        self.agent: Agent = Agent(self.environment)

    # def switch_to_target_policy(self):
    #     # if self.comparison.comparison_settings.dual_policy_relationship in \
    #     #     (common.DualPolicyRelationship.LINKED_POLICIES, common.DualPolicyRelationship.INDEPENDENT_POLICIES):
    #     self.agent.set_behaviour_policy(self.agent.target_policy)
    #
    # def update_grid_value_functions(self):
    #     policy_for_display = self.agent.policy.linked_policy
    #     self.environment.update_grid_value_functions(algorithm=self.agent.algorithm,
    #                                                  policy=policy_for_display)
    #
    # def _display_step(self, episode_: Optional[Episode]):
    #     # if self.trainer.episode_counter >= 9900:
    #     self.update_grid_value_functions()
    #     self._controller.display_step(episode_)
