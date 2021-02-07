import os_environ_settings
import common
import controller


def main():
    os_environ_settings.dummy = None
    controller_ = controller.Controller(verbose=False)
    controller_.setup_and_run(comparison_type=common.COMPARISON)
    controller_.run()


if __name__ == '__main__':
    main()
