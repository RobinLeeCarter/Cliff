import constants
import environment
import agent
from algorithm import episodic_algorithm


class VQ(episodic_algorithm.EpisodicAlgorithm):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 alpha: float = 0.5,
                 verbose: bool = False
                 ):
        title = f"VQ α={alpha}"
        super().__init__(environment_, agent_, title, verbose)
        self._alpha = alpha

    def _do_training_step(self):
        self.agent.choose_action()
        self.agent.take_action()
        sarsa = self.agent.get_sarsa()

        v_delta = sarsa.reward + \
            constants.GAMMA * self._V[sarsa.next_state] - \
            self._V[sarsa.state]
        self._V[sarsa.state] += self._alpha * v_delta

        q_delta = sarsa.reward + \
            constants.GAMMA * self._V[sarsa.next_state] - \
            self._Q[sarsa.state, sarsa.action]
        self._Q[sarsa.state, sarsa.action] += self._alpha * q_delta

        self.agent.policy[sarsa.state] = self._Q.argmax_over_actions(sarsa.state)
