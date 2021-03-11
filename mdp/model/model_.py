from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model import environment

import utils
from mdp import controller, common
from mdp.scenarios.factory import environment_factory
from mdp.model import breakdown, trainer, agent


class Model:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose
        self._controller: Optional[controller.Controller] = None
        self.comparison: Optional[common.Comparison] = None
        self.environment: Optional[environment.Environment] = None
        self.agent: Optional[agent.Agent] = None
        self.breakdown: Optional[breakdown.Breakdown] = None
        self.trainer: Optional[trainer.Trainer] = None

        self._cont: bool = True

    def set_controller(self, controller_: controller.Controller):
        self._controller = controller_

    def build(self, comparison: common.Comparison):
        self.comparison = comparison

        self.environment = environment_factory.environment_factory(self.comparison.environment_parameters)

        # create agent (and it will create the algorithm and the policy when it is given Settings)
        self.agent = agent.Agent(self.environment)

        self.breakdown: breakdown.Breakdown = breakdown.factory(self.comparison)
        self.trainer: trainer.Trainer = trainer.Trainer(
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
        for settings in self.comparison.settings_list:
            self.trainer.train(settings)
            if not self._cont:
                break
            timer.lap(name=str(settings.algorithm_title))
        timer.stop()

        self.breakdown.compile()
        # graph_values: common.GraphValues = self.breakdown.get_graph_values()

    def prep_for_output(self):
        self.environment.output_mode()
        self.switch_to_target_policy()
        self.update_grid_value_functions()

    def switch_to_target_policy(self):
        # if self.comparison.comparison_settings.dual_policy_relationship in \
        #     (common.DualPolicyRelationship.LINKED_POLICIES, common.DualPolicyRelationship.INDEPENDENT_POLICIES):
        self.agent.set_behaviour_policy(self.agent.target_policy)

    def update_grid_value_functions(self):
        policy_for_display = self.agent.policy.linked_policy
        self.environment.update_grid_value_functions(algorithm_=self.agent.algorithm, policy_=policy_for_display)

    def _display_step(self, episode_: agent.Episode):
        self.update_grid_value_functions()
        self._controller.display_step(episode_)
