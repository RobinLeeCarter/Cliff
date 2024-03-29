from mdp.task.mountain_car.model.environment_parameters import EnvironmentParameters
from mdp.task.mountain_car.model.environment import Environment
from mdp.task.mountain_car.model.state import State
from mdp.task.mountain_car.model.action import Action


def main():
    environment_parameters: EnvironmentParameters = EnvironmentParameters()
    environment: Environment = Environment(environment_parameters)
    environment.build()

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
