from __future__ import annotations
from typing import Optional

import utils
import common
from mdp.model import breakdown, trainer, scenarios, agent
from mdp import controller


class Model:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose
        self._controller: Optional[controller.Controller] = None
        self.comparison: Optional[common.Comparison] = None

    def build(self, comparison: common.Comparison):
        self.comparison = comparison

        self.environment = scenarios.environment_factory(self.comparison.environment_parameters)

        # create agent (and it will create the algorithm and the policy when it is given Settings)
        self.agent = agent.Agent(self.environment)

        self.breakdown: breakdown.Breakdown = breakdown.factory(self.comparison, self.graph)
        self.trainer: trainer.Trainer = trainer.Trainer(
            agent_=self.agent,
            breakdown_=self.breakdown,
            verbose=False
        )
        self.breakdown.set_trainer(self.trainer)

        # self.target_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.environment)
        # self.behaviour_policy: policy.EGreedyPolicy = policy.EGreedyPolicy(self.environment,
        #                                                                    greedy_policy=self.target_policy)
        # self.behaviour_policy: policy.RandomPolicy = policy.RandomPolicy(self.environment)
        # self.target_agent = agent.Agent(self.environment, self.target_policy)
        # self.behaviour_agent = agent.Agent(self.environment, self.behaviour_policy)

    def set_controller(self, controller_: controller.Controller):
        self._controller = controller_

    def run(self):
        timer: utils.Timer = utils.Timer()
        timer.start()
        for settings in self.comparison.settings_list:
            self.trainer.train(settings)
            timer.lap(name=str(settings.algorithm_title))
        timer.stop()

        self.breakdown.compile()
        self.breakdown.draw_graph()
