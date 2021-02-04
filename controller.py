from typing import Optional

import numpy as np

import utils
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

        self.factory: Optional[algorithm.Factory] = None
        self.algorithms: list[algorithm.EpisodicAlgorithm] = []
        self.algorithm_iteration_recorder: Optional[train.Recorder] = None
        self.trainer: Optional[train.Trainer] = None

        self.graph = view.Graph()

        self.grid_view = view.GridView(self.environment.grid_world)

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

    def setup(self, comparison: str):
        if comparison == "Return by episode":
            settings_list = data.single_alpha_comparison
            self.algorithms = [self.factory[settings_] for settings_ in settings_list]
            recorder_key_type: type = tuple[algorithm.EpisodicAlgorithm, int]
            self.algorithm_iteration_recorder: train.Recorder = train.Recorder[recorder_key_type]()
            self.trainer = train.Trainer(
                self.algorithm_iteration_recorder,
                verbose=False
            )
        elif comparison == "Return by alpha":
            pass


    def run(self):
        algorithms_output: dict[algorithm.EpisodicAlgorithm, np.ndarray] = {}

        timer: utils.Timer = utils.Timer()
        timer.start()
        for algorithm_ in self.algorithms:
            self.trainer.set_algorithm(algorithm_)
            self.trainer.train()
            self.algorithms_output[algorithm_] = self.trainer.return_array
            algorithm_.print_q_coverage_statistics()
            timer.lap(name=algorithm_.title)
        timer.stop()

        iteration_array = self.trainer.iteration_array
        self.graph.make_plot(iteration_array, algorithms_output, is_moving_average=False)

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
