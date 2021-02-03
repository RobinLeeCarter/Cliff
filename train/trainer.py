from typing import Optional

import numpy as np

import constants
import algorithm
from train import recorder


class Trainer:
    def __init__(self,
                 av_recorder_: recorder.Recorder,
                 algorithm_: algorithm.EpisodicAlgorithm = None,
                 verbose: bool = False
                 ):
        self.recorder: recorder.Recorder = av_recorder_
        self.algorithm: Optional[algorithm.EpisodicAlgorithm] = algorithm_
        # self.agent: agent.Agent = algorithm_.agent
        self.verbose = verbose

        # self.array_shape = (self.total_records, )
        # self.iteration_array = np.zeros(self.array_shape, dtype=int)
        # self.return_array = np.zeros(self.array_shape, dtype=float)

        self.iteration_array = np.array([], dtype=int)
        self.return_array = np.array([], dtype=float)

    def set_algorithm(self, algorithm_: algorithm.EpisodicAlgorithm):
        self.algorithm: algorithm.EpisodicAlgorithm = algorithm_
        print(algorithm_)

    def train(self):
        for run in range(constants.RUNS):
            print(f"run = {run}")
            self.algorithm.initialize()

            for iteration in range(constants.TRAINING_ITERATIONS):
                # print(f"iteration = {iteration}")
                if self.verbose:
                    print(f"iteration = {iteration}")
                # else:
                #     if iteration % 1000 == 0:
                #         print(f"iteration = {iteration}")

                self.algorithm.do_episode()
                max_t = self.algorithm.agent.t
                total_return = self.algorithm.agent.episode.total_return
                if self.verbose:
                    print(f"max_t = {max_t} \ttotal_return = {total_return:.2f}")
                if self.is_record_iteration(iteration):
                    self.recorder[self.algorithm, iteration] = total_return

                # if iteration > 20:
                #     for test in range(constants.TESTS):
                #         self.agent.generate_episode()
                #         # max_t = self.algorithm.agent.t
                #         total_return = self.agent.episode.total_return
                #         if self.is_record_iteration(iteration):
                #             self.recorder[self.algorithm, iteration] = total_return

            # print(self.recorder.tallies)

        self.fill_arrays_from_recorder()

    def fill_arrays_from_recorder(self):
        iteration: int = 0
        iteration_list: list[int] = []
        return_list: list[float] = []

        while iteration < constants.TRAINING_ITERATIONS:
            if self.is_record_iteration(iteration):
                total_return = self.recorder[self.algorithm, iteration]
                iteration_list.append(iteration)
                return_list.append(total_return)
            iteration += 1

        self.iteration_array = np.array(iteration_list)
        self.return_array = np.array(return_list)

    # @property
    # def total_records(self) -> int:
    #     start = math.ceil(constants.PERFORMANCE_SAMPLE_START / constants.PERFORMANCE_SAMPLE_FREQUENCY)
    #     stop = math.floor(constants.TRAINING_ITERATIONS / constants.PERFORMANCE_SAMPLE_FREQUENCY)
    #     return stop - start + 1

    def is_record_iteration(self, iteration: int) -> bool:
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
