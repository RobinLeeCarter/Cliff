from __future__ import annotations

import os_environ_settings
import model


def main():
    os_environ_settings.dummy = None    # for pycharm code inspection only
    model_ = model.Model(verbose=False)
    model_.run()
    # model_.demonstrate()


if __name__ == '__main__':
    main()
