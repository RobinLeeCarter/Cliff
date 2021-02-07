import data
import environment
import view


def view_test() -> bool:
    grid_world_ = environment.GridWorld(data.CLIFF_GRID)
    my_view = view.GridView(grid_world_)
    my_view.open_window()
    my_view.display_and_wait()

    return True


if __name__ == '__main__':
    if view_test():
        print("Passed")
