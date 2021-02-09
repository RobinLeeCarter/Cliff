import environment
import policy
import agent
import algorithm
import train
import data


def recorder_test() -> bool:
    environment_ = environment.Environment(data.CLIFF_GRID, verbose=False)
    greedy_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(environment_)
    e_greedy_policy: policy.EGreedyPolicy = policy.EGreedyPolicy(environment_, greedy_policy=greedy_policy)
    agent_ = agent.Agent(environment_, e_greedy_policy)

    factory_ = algorithm.Factory(environment_, agent_)
    algorithms: list[algorithm.Episodic] = [factory_[settings_] for settings_ in data.return_by_episode_settings]
    av_recorder_ = train.Recorder()

    algorithm_ = algorithms[0]
    print(algorithm_)
    print(algorithm_.__hash__())
    av_recorder_[algorithm_, 0] = 12.0
    print(av_recorder_[algorithm_, 0])

    return True


if __name__ == '__main__':
    if recorder_test():
        print("Passed")
