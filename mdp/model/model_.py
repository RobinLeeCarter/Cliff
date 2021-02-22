from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model import environment
    # from mdp.model.algorithm import value_function

import utils
from mdp import controller, common
from mdp.model import breakdown, trainer, scenarios, agent, policy


class Model:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose
        self._controller: Optional[controller.Controller] = None
        self.comparison: Optional[common.Comparison] = None
        self.environment: Optional[environment.Environment] = None
        self.agent: Optional[agent.Agent] = None
        self.breakdown: Optional[breakdown.Breakdown] = None
        self.trainer: Optional[trainer.Trainer] = None

    def set_controller(self, controller_: controller.Controller):
        self._controller = controller_

    def build(self, comparison: common.Comparison):
        self.comparison = comparison

        self.environment = scenarios.environment_factory(self.comparison.environment_parameters)

        # create agent (and it will create the algorithm and the policy when it is given Settings)
        self.agent = agent.Agent(self.environment)

        self.breakdown: breakdown.Breakdown = breakdown.factory(self.comparison)
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

    def run(self):
        timer: utils.Timer = utils.Timer()
        timer.start()
        for settings in self.comparison.settings_list:
            self.trainer.train(settings)
            timer.lap(name=str(settings.algorithm_title))
        timer.stop()

        self.breakdown.compile()
        # graph_values: common.GraphValues = self.breakdown.get_graph_values()

    def update_grid_value_functions(self):
        policy_ = self.agent.policy
        assert isinstance(policy_, policy.EGreedy)
        greedy_policy = policy_.greedy_policy

        for state in self.environment.states():
            policy_action: Optional[environment.Action] = greedy_policy[state]
            policy_move: Optional[common.XY] = None
            if policy_action:
                policy_move = policy_action.move

            self.environment.grid_world.set_state_function(
                position=state.position,
                v_value=self.agent.algorithm.V[state]
            )
            for action in self.environment.actions_for_state(state):
                is_policy: bool = (policy_move and policy_move == action.move)
                self.environment.grid_world.set_state_action_function(
                    position=state.position,
                    move=action.move,
                    q_value=self.agent.algorithm.Q[state, action],
                    is_policy=is_policy
                )
        # print(self.environment.grid_world.output_squares)
