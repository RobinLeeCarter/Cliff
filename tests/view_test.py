from __future__ import annotations

import environments
import view


def view_test() -> bool:
    cliff = environments.Cliff()
    my_view = view.GridView(cliff.grid_world)
    my_view.open_window()
    my_view.display_and_wait()

    return True


if __name__ == '__main__':
    if view_test():
        print("Passed")
