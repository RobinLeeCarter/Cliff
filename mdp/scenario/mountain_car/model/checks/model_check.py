from mdp.scenario.mountain_car.model.model import Model
from mdp.scenario.mountain_car.model.environment_parameters import EnvironmentParameters
from mdp.scenario.mountain_car.model.environment import Environment
from mdp.scenario.mountain_car.model.state import State
from mdp.scenario.mountain_car.model.action import Action


def main():
    environment_parameters: EnvironmentParameters = EnvironmentParameters()
    model: Model = Model()

    # noinspection PyProtectedMember
    environment: Environment = model._create_environment(environment_parameters)
    environment.build()
    model.environment = environment

    # noinspection PyProtectedMember
    model.agent = model._create_agent()

    model.agent.state = State(is_terminal=False, position=1.0, velocity=2.0)

    state: State = environment.draw_start_state()
    # state = State(is_terminal=False, position=0.1, velocity=0.3)
    action = Action(acceleration=1.0)

    print(state)
    print(action)

    for i in range(1000):
        reward, state = environment.from_state_perform_action(state, action)
        print(f"{i}: \tposition={state.position:.2f} \tvelocity={state.velocity:.2f}")
        if state.is_terminal:
            break

    # print(state.values)
    # print(action.values)


if __name__ == "__main__":
    main()
