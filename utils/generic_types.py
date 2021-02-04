import typing


def get_generic_types(generic_instance) -> tuple:
    return typing.get_args(generic_instance.__orig_class__)
