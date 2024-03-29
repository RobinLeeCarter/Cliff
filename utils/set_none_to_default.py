from __future__ import annotations


def set_none_to_default(obj: object, default: object):
    for attribute_name, attribute_value in vars(obj).items():
        if attribute_value is None:
            default_value = default.__getattribute__(attribute_name)
            # print(f"{attribute_name=}\t\t{attribute_value=}\t\t{default_value=}")
            obj.__setattr__(attribute_name, default_value)
