from environment.grid_world import grid_world
import data
import view


def view_test() -> bool:
    grid_world_ = grid_world.GridWorld(data.CLIFF_GRID)
    my_view = view.GridView(grid_world_)
    my_view.open_window()
    my_view.display_and_wait()

    return True


if __name__ == '__main__':
    if view_test():
        print("Passed")
