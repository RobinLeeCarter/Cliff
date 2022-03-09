from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
    # from mdp.model.tabular.agent.episode import Episode
    from mdp.model.breakdown.breakdown import Breakdown

import multiprocessing
import utils
from mdp import common
# from mdp.scenarios.factory import environment_factory
from mdp.model.tabular.agent.agent import Agent
from mdp.model.breakdown import breakdown_factory
from mdp.model.trainer.trainer import Trainer
from mdp.model.trainer.parallel_trainer import ParallelTrainer

from mdp.model.model import Model


class TabularModel(Model):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.environment: Optional[TabularEnvironment] = None
        self.agent: Optional[Agent] = None
        # self.breakdown: Optional[Breakdown] = None
        # self.trainer: Optional[Trainer] = None
        # self.parallel_trainer: Optional[ParallelTrainer] = None

    def _build(self):
        # different for each scenario and environment_parameters
        self.environment: TabularEnvironment = self._create_environment(self._comparison.environment_parameters)
        self.environment.build()
        # self.environment = environment_factory.environment_factory(self._comparison.environment_parameters)

        # create agent (and it will create the algorithm and the policy when it is given Settings)
        self.agent = Agent(self.environment)

        # breakdowns themselves need comparison in current implementation so breakdown_parameters is not passed in
        self.breakdown: Optional[Breakdown] = breakdown_factory.breakdown_factory(self._comparison)
        self.trainer: Trainer = Trainer(
            agent_=self.agent,
            breakdown_=self.breakdown,
            model_step_callback=self._display_step,
            verbose=False
        )
        if self.breakdown:
            self.breakdown.set_trainer(self.trainer)
        if self._comparison.settings_list_multiprocessing != common.ParallelContextType.NONE:
            self.parallel_trainer = ParallelTrainer(self.trainer, self._comparison.settings_list_multiprocessing)

    @abstractmethod
    def _create_environment(self, environment_parameters: common.EnvironmentParameters) -> TabularEnvironment:
        pass

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
