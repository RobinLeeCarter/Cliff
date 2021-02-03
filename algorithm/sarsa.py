import constants
import environment
import agent
from algorithm import episodic_algorithm


class Sarsa(episodic_algorithm.EpisodicAlgorithm):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 alpha: float = 0.5,
                 verbose: bool = False
                 ):
        title = f"SARSA alpha={alpha}"
        super().__init__(environment_, agent_, title, verbose)
        self._alpha = alpha

    def _do_training_step(self):
        self.agent.take_action()
        sarsa = self.agent.get_sarsa()
        delta = sarsa.reward + \
            constants.GAMMA * self._Q[sarsa.next_state, sarsa.next_action] - \
            self._Q[sarsa.state, sarsa.action]
        self._Q[sarsa.state, sarsa.action] += self._alpha * delta
        self.agent.policy[sarsa.state] = self._Q.argmax_over_actions(sarsa.state)
