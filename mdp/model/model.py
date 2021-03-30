from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.episode import Episode
    from mdp.model.breakdown.breakdown import Breakdown

import utils
from mdp import controller, common
from mdp.scenarios.factory import environment_factory
from mdp.model.agent.agent import Agent
from mdp.model.breakdown import factory
from mdp.model.trainer import Trainer


class Model:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose
        self._controller: Optional[controller.Controller] = None
        self._comparison: Optional[common.Comparison] = None
        self.environment: Optional[Environment] = None
        self.agent: Optional[Agent] = None
        self.breakdown: Optional[Breakdown] = None
        self.trainer: Optional[Trainer] = None

        self._cont: bool = True

    def set_controller(self, controller_: controller.Controller):
        self._controller = controller_

    def build(self, comparison: common.Comparison):
        self._comparison = comparison

        self.environment = environment_factory.environment_factory(self._comparison.environment_parameters)

        # create agent (and it will create the algorithm and the policy when it is given Settings)
        self.agent = Agent(self.environment)

        self.breakdown: Optional[Breakdown] = factory.breakdown_factory(self._comparison)
        self.trainer: Trainer = Trainer(
            agent_=self.agent,
            breakdown_=self.breakdown,
            model_step_callback=self._display_step,
            verbose=False
        )
        if self.breakdown:
            self.breakdown.set_trainer(self.trainer)

        # self.target_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.environment)
        # self.behaviour_policy: policy.EGreedyPolicy = policy.EGreedyPolicy(self.environment,
        #                                                                    greedy_policy=self.target_policy)
        # self.behaviour_policy: policy.RandomPolicy = policy.RandomPolicy(self.environment)
        # self.target_agent = agent.Agent(self.environment, self.target_policy)
        # self.behaviour_agent = agent.Agent(self.environment, self.behaviour_policy)

    def run(self):
        timer: utils.Timer = utils.Timer()
        timer.start()
        for settings in self._comparison.settings_list:
            self.trainer.train(settings)
            if not self._cont:
                break
            timer.lap(name=str(settings.algorithm_title))
        timer.stop()

        if self.breakdown:
            self.breakdown.compile()
        # graph_values: common.GraphValues = self.breakdown.get_graph_values()

    def prep_for_output(self, parameter: any = None):
        self.environment.output_mode()
        self.switch_to_target_policy()
        self.update_grid_value_functions(parameter)

    def switch_to_target_policy(self):
        # if self.comparison.comparison_settings.dual_policy_relationship in \
        #     (common.DualPolicyRelationship.LINKED_POLICIES, common.DualPolicyRelationship.INDEPENDENT_POLICIES):
        self.agent.set_behaviour_policy(self.agent.target_policy)

    def update_grid_value_functions(self, parameter: any = None):
        policy_for_display = self.agent.policy.linked_policy
        self.environment.update_grid_value_functions(algorithm_=self.agent.algorithm,
                                                     policy_=policy_for_display,
                                                     parameter=parameter)

    def _display_step(self, episode_: Optional[Episode]):
        self.update_grid_value_functions()
        self._controller.display_step(episode_)
