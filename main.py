from __future__ import annotations

import os_environ_settings
import controller


def main():
    os_environ_settings.dummy = None    # for pycharm code inspection only
    controller_ = controller.Controller(verbose=False)
    controller_.run()
    # controller_.demonstrate()


if __name__ == '__main__':
    main()
