import numpy as np

import comparison
import algorithm


class Trainer:
    def __init__(self,
                 algorithm_factory: algorithm.Factory,
                 comparison_: comparison.Comparison,
                 verbose: bool = False
                 ):
        self.algorithm_factory: algorithm.Factory = algorithm_factory
        self.comparison: comparison.Comparison = comparison_
        self.verbose = verbose

    def train(self, settings: comparison.Settings):
        algorithm_ = self.algorithm_factory[settings]
        print(algorithm_)

        for run in range(settings.runs):
            if self.verbose or run % settings.run_print_frequency == 0:
                print(f"run = {run}")
            algorithm_.initialize()

            for iteration in range(settings.training_iterations):
                algorithm_.parameter_changes(iteration)
                # print(f"iteration = {iteration}")
                if self.verbose or iteration % settings.iteration_print_frequency == 0:
                    print(f"iteration = {iteration}")

                algorithm_.do_episode(settings.episode_length_timeout)
                # max_t = algorithm_.agent.t
                # total_return = algorithm_.agent.episode.total_return
                # if self.verbose:
                #     print(f"max_t = {max_t} \ttotal_return = {total_return:.2f}")
                self.comparison.review(settings, iteration, algorithm_.agent.episode)

                # if self.is_record_iteration(iteration):
                #     # if isinstance(self.recorder, algorithm_iteration_recorder.AlgorithmIterationRecorder):
                #     # if self.recorder.key_type == tuple[algorithm.EpisodicAlgorithm, int]:
                #     if self.comparison == common.Comparison.RETURN_BY_EPISODE:
                #         self.recorder[algorithm_, iteration] = total_return
                #     elif self.comparison == common.Comparison.RETURN_BY_ALPHA:
                #         self.recorder[algorithm_type, alpha] = total_return

                # if iteration > 20:
                #     for test in range(constants.TESTS):
                #         self.agent.generate_episode()
                #         # max_t = self.algorithm.agent.t
                #         total_return = self.agent.episode.total_return
                #         if self.is_record_iteration(iteration):
                #             self.recorder[self.algorithm, iteration] = total_return

            # print(self.recorder.tallies)
        algorithm_.print_q_coverage_statistics()

        # self.fill_arrays_from_recorder()

    # def fill_arrays_from_recorder(self):
    #     """This should be outside of Trainer"""
    #     iteration: int = 0
    #     iteration_list: list[int] = []
    #     return_list: list[float] = []
    #
    #     while iteration < constants.TRAINING_ITERATIONS:
    #         if self.is_record_iteration(iteration):
    #             if self.recorder.key_type == tuple[algorithm.EpisodicAlgorithm, int]:
    #                 total_return = self.recorder[self.algorithm, iteration]
    #             else:
    #                 total_return = None
    #             iteration_list.append(iteration)
    #             return_list.append(total_return)
    #         iteration += 1
    #
    #     self.iteration_array = np.array(iteration_list)
    #     self.return_array = np.array(return_list)

    # @property
    # def total_records(self) -> int:
    #     start = math.ceil(constants.PERFORMANCE_SAMPLE_START / constants.PERFORMANCE_SAMPLE_FREQUENCY)
    #     stop = math.floor(constants.TRAINING_ITERATIONS / constants.PERFORMANCE_SAMPLE_FREQUENCY)
    #     return stop - start + 1

    # def is_record_iteration(self, iteration: int, settings_: comparison.Settings) -> bool:
    #     return iteration >= self.sett .PERFORMANCE_SAMPLE_START and \
    #             iteration % constants.PERFORMANCE_SAMPLE_FREQUENCY == 0

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
