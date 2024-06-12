"""
    Handy misc functions
"""


def check_type(item, item_name, desired_type):
    """
    Check that item is of desired type
    """

    if not isinstance(item, (desired_type)):
        raise TypeError(
            f"{item_name} must be given as {desired_type}, got {type(item)}")
