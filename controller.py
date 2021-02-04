import numpy as np

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

        recorder_key_type: type = tuple[algorithm.EpisodicAlgorithm, int]
        self.algorithm_iteration_recorder = train.Recorder[recorder_key_type]()
        print(self.algorithm_iteration_recorder.key_type)

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

    def run(self):
        algorithms_output: dict[algorithm.EpisodicAlgorithm, np.ndarray] = {}
        # sarsa_array: np.ndarray = np.array([], dtype=float)
        trainer: train.Trainer = train.Trainer(
            self.algorithm_iteration_recorder,
            verbose=False
        )
        for algorithm_ in self.algorithms:
            trainer.set_algorithm(algorithm_)
            trainer.train()
            algorithms_output[algorithm_] = trainer.return_array
            algorithm_.print_q_coverage_statistics()

        iteration_array = trainer.iteration_array
        self.graph.ma_plot(iteration_array, algorithms_output)

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
