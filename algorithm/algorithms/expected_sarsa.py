import constants
import environment
import agent
from algorithm.algorithms import episodic_algorithm


class ExpectedSarsa(episodic_algorithm.EpisodicAlgorithm):
    name: str = "Expected Sarsa"

    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 alpha: float = 0.5,
                 verbose: bool = False
                 ):
        title = f"{ExpectedSarsa.name} α={alpha}"
        super().__init__(environment_, agent_, title, verbose)
        self._alpha = alpha

    def _do_training_step(self):
        self.agent.choose_action()
        self.agent.take_action()
        sarsa = self.agent.get_sarsa()
        q_expectation_over_a = self._get_expectation_over_a(sarsa.state)
        delta = sarsa.reward \
            + constants.GAMMA * q_expectation_over_a \
            - self._Q[sarsa.prev_state, sarsa.prev_action]
        self._Q[sarsa.prev_state, sarsa.prev_action] += self._alpha * delta
        self.agent.policy[sarsa.prev_state] = self._Q.argmax_over_actions(sarsa.prev_state)

    def _get_expectation_over_a(self, state: environment.State) -> float:
        expectation: float = 0.0
        for action in self.environment.actions():
            probability = self.agent.policy.get_probability(state, action)
            expectation += probability * self._Q[state, action]
        return expectation
