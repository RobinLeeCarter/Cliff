import constants
import algorithm
from train import recorder


class Trainer:
    def __init__(self,
                 av_recorder_: recorder.Recorder,
                 algorithm_: algorithm.EpisodicAlgorithm,
                 verbose: bool = False
                 ):
        self.av_recorder: recorder.Recorder = av_recorder_
        self.algorithm: algorithm.EpisodicAlgorithm = algorithm_
        self.verbose = verbose

        self.learning_iteration: int = 0

        # samples = int(constants.LEARNING_EPISODES / constants.PERFORMANCE_SAMPLE_FREQUENCY)
        # self.sample_iteration: np.ndarray = np.zeros(shape=samples+1, dtype=int)
        # self.average_return: np.ndarray = np.zeros(shape=samples+1, dtype=float)
        # self.average_return[0] = constants.INITIAL_Q_VALUE*2

    def train(self):
        self.learning_iteration = 0

        while self.learning_iteration < constants.LEARNING_EPISODES:
            if self.verbose:
                print(f"iteration = {self.learning_iteration}")
            else:
                if self.learning_iteration % 1000 == 0:
                    print(f"iteration = {self.learning_iteration}")

            self.algorithm.do_episode()

            max_t = self.algorithm.agent.t
            total_return = self.algorithm.agent.episode.total_return
            if self.verbose:
                print(f"max_t = {max_t} \ttotal_return = {total_return:.2f}")

            if self.learning_iteration >= constants.PERFORMANCE_SAMPLE_START and \
                    self.learning_iteration % constants.PERFORMANCE_SAMPLE_FREQUENCY == 0:
                self.av_recorder[self.algorithm, self.learning_iteration] = total_return

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
