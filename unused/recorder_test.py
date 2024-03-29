from mdp.model import algorithm
from mdp.model import scenarios, policy, agent
from mdp.model.breakdown import recorder


def recorder_test() -> bool:
    environment_ = scenarios.TabularEnvironment()
    greedy_policy: policy.Deterministic = policy.Deterministic(environment_)
    e_greedy_policy: policy.EGreedy = policy.EGreedy(environment_, greedy_policy=greedy_policy)
    agent_ = agent.TabularAgent(environment_, e_greedy_policy)

    factory_ = algorithm.Factory(environment_, agent_)
    algorithms: list[algorithm.Algorithm] = [factory_[settings_] for settings_ in data.return_by_episode_settings]
    av_recorder_ = recorder.Recorder()

    algorithm_ = algorithms[0]
    print(algorithm_)
    print(algorithm_.__hash__())
    av_recorder_[algorithm_, 0] = 12.0
    print(av_recorder_[algorithm_, 0])

    return True


if __name__ == '__main__':
    if recorder_test():
        print("Passed")
