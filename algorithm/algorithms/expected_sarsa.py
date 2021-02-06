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

        prev_state = self.agent.prev_state
        prev_action = self.agent.action
        reward = self.agent.reward
        state = self.agent.state

        q_expectation_over_a = self._get_expectation_over_a(state)
        delta = reward \
            + constants.GAMMA * q_expectation_over_a \
            - self._Q[prev_state, prev_action]
        self._Q[prev_state, prev_action] += self._alpha * delta
        self.agent.policy[prev_state] = self._Q.argmax_over_actions(prev_state)

    def _get_expectation_over_a(self, state: environment.State) -> float:
        expectation: float = 0.0
        for action in self.environment.actions():
            probability = self.agent.policy.get_probability(state, action)
            expectation += probability * self._Q[state, action]
        return expectation
