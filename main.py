import os_environ_settings
import common
import controller


def main():
    os_environ_settings.dummy = None
    controller_ = controller.Controller(verbose=False)
    controller_.setup_and_run(comparison=common.Comparison.RETURN_BY_EPISODE)
    controller_.run()


if __name__ == '__main__':
    main()
