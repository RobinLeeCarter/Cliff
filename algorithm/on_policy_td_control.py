import constants
import environment
import policy
import agent
from algorithm import state_action_function


class OnPolicyTdControl:
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 gamma: float = 1.0,
                 alpha: float = 0.5,
                 verbose: bool = False
                 ):
        self.environment: environment.Environment = environment_
        self.agent: agent.Agent = agent_
        assert isinstance(self.agent.policy, policy.EGreedyPolicy), "agent.policy not EGreedyPolicy"
        self.policy: policy.EGreedyPolicy = self.agent.policy
        self.gamma = gamma
        self.alpha = alpha
        self.verbose = verbose

        self.Q = state_action_function.StateActionFunction(self.environment)
        self.initialise_policy()
        self.learning_iteration: int = 0

        # samples = int(constants.LEARNING_EPISODES / constants.PERFORMANCE_SAMPLE_FREQUENCY)
        # self.sample_iteration: np.ndarray = np.zeros(shape=samples+1, dtype=int)
        # self.average_return: np.ndarray = np.zeros(shape=samples+1, dtype=float)
        # self.average_return[0] = constants.INITIAL_Q_VALUE*2

    def initialise_policy(self):
        for state_ in self.environment.states():
            self.policy[state_] = self.Q.argmax_over_actions(state_)

    # noinspection PyPep8Naming
    def run(self):
        self.learning_iteration: int = 0
        while self.learning_iteration <= constants.LEARNING_EPISODES:
            if self.verbose:
                print(f"iteration = {self.learning_iteration}")
            else:
                if self.learning_iteration % 1000 == 0:
                    print(f"iteration = {self.learning_iteration}")
            # if self.learning_iteration >= constants.PERFORMANCE_SAMPLE_START and \
            #         self.learning_iteration % constants.PERFORMANCE_SAMPLE_FREQUENCY == 0:
            #     self.sample_target()

            self.agent.start_episode()
            while (not self.agent.state.is_terminal) and self.agent.t <= 10000:
                self.agent.take_action()
                sarsa = self.agent.get_sarsa()
                delta = sarsa.reward + \
                    self.gamma * self.Q[sarsa.next_state, sarsa.next_action] - \
                    self.Q[sarsa.state, sarsa.action]
                self.Q[sarsa.state, sarsa.action] += self.alpha * delta
                self.agent.policy[sarsa.state] = self.Q.argmax_over_actions(sarsa.state)
            if self.verbose:
                print(f"max_t = {self.agent.t}")
            self.learning_iteration += 1

        # print(self.sample_iteration)
        # print(self.average_return)

    # noinspection PyPep8Naming
    # def sample_target(self):
    #     sample_number = int(self.learning_iteration / constants.PERFORMANCE_SAMPLE_FREQUENCY)
    #     # print(sample_number)
    #     sample_iteration: int = 1
    #     average_G: float = 0.0
    #     while sample_iteration <= constants.PERFORMANCE_SAMPLES:
    #         if self.verbose:
    #             print(f"sample_iteration = {sample_iteration}")
    #
    #         episode: agent.Episode = self.target_agent.generate_episode()
    #         G: float = 0.0
    #         for reward_state_action in reversed(episode.trajectory):
    #             if reward_state_action.reward is not None:
    #                 G = self.gamma * G + reward_state_action.reward
    #         average_G += (1/sample_iteration) * (G - average_G)
    #         sample_iteration += 1
    #     self.sample_iteration[sample_number] = self.learning_iteration
    #     self.average_return[sample_number] = average_G
