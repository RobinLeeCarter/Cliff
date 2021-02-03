import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure

import constants
import environment
import policy
import agent
import algorithm
import train
import view
import data


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose

        self.rng: np.random.Generator = np.random.default_rng()
        self.environment = environment.Environment(data.GRID_1, self.rng, verbose=False)
        self.greedy_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.environment)
        self.e_greedy_policy: policy.EGreedyPolicy = policy.EGreedyPolicy(self.environment, self.rng,
                                                                          greedy_policy=self.greedy_policy)
        self.agent = agent.Agent(self.environment, self.e_greedy_policy)

        self.factory = algorithm.Factory(self.environment, self.agent)
        self.algorithms: list[algorithm.EpisodicAlgorithm] =\
            [self.factory[settings_] for settings_ in data.SETTINGS_LIST]
        # print(self.algorithms)
        self.av_recorder = train.Recorder()

        self.view = view.View(self.environment.grid_world)

        # self.target_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.environment)
        # self.behaviour_policy: policy.EGreedyPolicy = policy.EGreedyPolicy(self.environment, self.rng,
        #                                                                    greedy_policy=self.target_policy)
        # self.behaviour_policy: policy.RandomPolicy = policy.RandomPolicy(self.environment, self.rng)
        # self.target_agent = agent.Agent(self.environment, self.target_policy)
        # self.behaviour_agent = agent.Agent(self.environment, self.behaviour_policy)

        # self.algorithm_: algorithm.EpisodicAlgorithm = algorithm.Sarsa(
        #         self.environment,
        #         self.agent,
        #         alpha=constants.ALPHA,
        #         verbose=False
        #     )

    def run(self):
        iteration_array: np.ndarray = np.array([], dtype=int)
        algorithms_output: dict[algorithm.EpisodicAlgorithm, np.ndarray] = {}
        # sarsa_array: np.ndarray = np.array([], dtype=float)
        trainer: train.Trainer = train.Trainer(
            self.av_recorder,
            verbose=False
        )
        for algorithm_ in self.algorithms:
            trainer.set_algorithm(algorithm_)
            trainer.train()
            algorithms_output[algorithm_] = trainer.return_array
            algorithm_.print_q_coverage_statistics()

        iteration_array = trainer.iteration_array
        # self.output_q()
        self.graph_samples(iteration_array, algorithms_output)

        # self.behaviour_agent.set_policy(self.target_policy)
        # self.view.open_window()
        # self.view.display_and_wait()
        # self.environment.verbose = True
        # self.agent.verbose = True

        # self.agent.verbose = True
        # self.agent.set_policy(self.greedy_policy)
        # while True:
        #     self.agent.generate_episode()
        #     print(f"max_t: {self.agent.episode.max_t}")
        #     user_event: common.UserEvent = self.view.display_episode(self.agent.episode, show_trail=False)
        #     if user_event == common.UserEvent.QUIT:
        #         break

    # def output_q(self):
    #     q = self.algorithm_.Q
    #     q_size = q.size
    #     q_non_zero = np.count_nonzero(q)
    #     percent_non_zero = 100.0 * q_non_zero / q_size
    #     print(f"q_size: {q_size}\tq_non_zero: {q_non_zero}\tpercent_non_zero: {percent_non_zero:.2f}")

    def graph_samples(self, iteration: np.ndarray, algorithms_output: dict[algorithm.EpisodicAlgorithm, np.ndarray]):
        fig: figure.Figure = plt.figure()
        ax: figure.Axes = fig.subplots()
        ax.set_title("Average Return vs Learning Episodes")
        ax.set_xlim(xmin=0, xmax=constants.TRAINING_ITERATIONS)
        ax.set_xlabel("Learning Episodes")
        ax.set_ylim(ymin=-100, ymax=0)
        ax.set_ylabel("Average Return")
        for algorithm_, return_array in algorithms_output.items():
            ax.plot(iteration, return_array, label=algorithm_.title)
        ax.legend()
        ax.grid(True)
        plt.show()
