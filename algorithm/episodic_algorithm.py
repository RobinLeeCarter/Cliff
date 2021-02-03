import abc

import constants
import environment
import agent
from algorithm import state_action_function


class EpisodicAlgorithm(abc.ABC):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 title: str,
                 verbose: bool = False
                 ):
        self.environment: environment.Environment = environment_
        self.agent: agent.Agent = agent_
        self.title: title = title
        self.verbose = verbose

        # assume all episodic algorithms have a Q function and initialise policy based on it
        # could add another layer of inheritance if not
        self._Q = state_action_function.StateActionFunction(self.environment)

    def initialize(self):
        self._Q.initialize_values()
        self._make_policy_greedy_wrt_q()

    def _make_policy_greedy_wrt_q(self):
        for state_ in self.environment.states():
            self.agent.policy[state_] = self._Q.argmax_over_actions(state_)

    def do_episode(self):
        self.agent.start_episode()
        while (not self.agent.state.is_terminal) and self.agent.t < constants.EPISODE_LENGTH_TIMEOUT:
            self._do_training_step()

    @abc.abstractmethod
    def _do_training_step(self):
        pass

    def print_q_coverage_statistics(self):
        self._Q.print_coverage_statistics()
