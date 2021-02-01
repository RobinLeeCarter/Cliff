import numpy as np

from environment import grid
import constants
import view

rng: np.random.Generator = np.random.default_rng()


def view_test() -> bool:
    grid_world = grid.GridWorld(constants.GRID, rng)
    my_view = view.View(grid_world)
    my_view.open_window()
    my_view.display_and_wait()

    return True


if __name__ == '__main__':
    if view_test():
        print("Passed")
