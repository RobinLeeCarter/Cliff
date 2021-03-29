from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import abc

if TYPE_CHECKING:
    from mdp.model import environment, agent
    from mdp.model.policy.policy import Policy
    from mdp import common
from mdp.model.algorithm import value_function


class Algorithm(abc.ABC):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        self._environment: environment.Environment = environment_
        self._agent: agent.Agent = agent_
        self._algorithm_parameters: common.AlgorithmParameters = algorithm_parameters
        self._policy_parameters: common.PolicyParameters = policy_parameters
        self._verbose = self._algorithm_parameters.verbose

        self._algorithm_type: Optional[common.AlgorithmType] = None
        self.name: str = "Error: Untitled"
        self.title: str = "Error: Untitled"

        self._gamma: float = self._agent.gamma
        self.V: Optional[value_function.StateFunction] = None
        self.Q: Optional[value_function.StateActionFunction] = None

    def _create_v(self):
        self.V = value_function.StateFunction(self._environment, self._algorithm_parameters.initial_v_value)

    def _create_q(self):
        self.Q = value_function.StateActionFunction(self._environment, self._algorithm_parameters.initial_q_value)

    def initialize(self):
        if self.V:
            self.V.initialize_values()
        if self.Q:
            self.Q.initialize_values()

    def parameter_changes(self, iteration: int):
        pass

    def _make_policy_greedy_wrt_q(self):
        for state_ in self._environment.states:
            if not state_.is_terminal:
                # works for single policy or dual policies
                self._agent.target_policy[state_] = self.Q.argmax_over_actions(state_)

    def print_q_coverage_statistics(self):
        self.Q.print_coverage_statistics()

    def __repr__(self):
        return f"{self.title}"

    def derive_v_from_q(self, policy_: Optional[Policy] = None):
        if not policy_:
            policy_ = self._agent.policy

        if not self.V:
            self._create_v()

        for state in self._environment.states:
            # Sum_over_a( π(a|s).Q(s,a) )
            expected_v: float = 0.0
            for action in self._environment.actions_for_state(state):
                # π(a|s)
                policy_probability = policy_.get_probability(state, action)
                # π(a|s).Q(s,a)
                expected_v += policy_probability * self.Q[state, action]
            self.V[state] = expected_v
