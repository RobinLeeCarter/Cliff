def get_enum_str(enum: object, value: int) -> str:
    for attribute_name, attribute_value in vars(enum).items():
        if attribute_value == value:
            return attribute_name
        else:
            return ""
