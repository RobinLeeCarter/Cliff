import numpy as np
import math

import constants
import algorithm
from train import recorder


class Trainer:
    def __init__(self,
                 av_recorder_: recorder.Recorder,
                 algorithm_: algorithm.EpisodicAlgorithm,
                 verbose: bool = False
                 ):
        self.recorder: recorder.Recorder = av_recorder_
        self.algorithm: algorithm.EpisodicAlgorithm = algorithm_
        self.verbose = verbose

        self.iteration: int = 0

        self.array_shape = (self.total_records, )
        self.iteration_array: np.zeros(self.array_shape, dtype=int)
        self.return_array: np.zeros(self.array_shape, dtype=float)

    def train(self):
        self.iteration = 0

        while self.iteration < constants.TRAINING_ITERATIONS:
            if self.verbose:
                print(f"iteration = {self.iteration}")
            else:
                if self.iteration % 1000 == 0:
                    print(f"iteration = {self.iteration}")

            self.algorithm.do_episode()

            max_t = self.algorithm.agent.t
            total_return = self.algorithm.agent.episode.total_return
            if self.verbose:
                print(f"max_t = {max_t} \ttotal_return = {total_return:.2f}")

            if self.is_record(self.iteration):
                self.recorder[self.algorithm, self.iteration] = total_return

            self.iteration += 1

    @property
    def total_records(self) -> int:
        start = math.ceil(constants.PERFORMANCE_SAMPLE_START / constants.PERFORMANCE_SAMPLE_FREQUENCY)
        stop = math.floor(constants.TRAINING_ITERATIONS / constants.PERFORMANCE_SAMPLE_FREQUENCY)
        return stop - start + 1

    def is_record(self, iteration: int) -> bool:
        return iteration >= constants.PERFORMANCE_SAMPLE_START and \
                iteration % constants.PERFORMANCE_SAMPLE_FREQUENCY == 0

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
