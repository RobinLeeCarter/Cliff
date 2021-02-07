import numpy as np

from environment import grid_world
import data
import view

rng: np.random.Generator = np.random.default_rng()


def view_test() -> bool:
    grid_world = grid_world.GridWorld(data.CLIFF_GRID, rng)
    my_view = view.GridView(grid_world)
    my_view.open_window()
    my_view.display_and_wait()

    return True


if __name__ == '__main__':
    if view_test():
        print("Passed")
