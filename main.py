import os_environ_settings
import controller


def main():
    os_environ_settings.dummy = None
    controller_ = controller.Controller(verbose=False)
    controller_.setup(comparison="Return by episode")
    controller_.run()


if __name__ == '__main__':
    main()
