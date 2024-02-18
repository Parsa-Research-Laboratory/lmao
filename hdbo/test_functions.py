from typing import Callable

VALID_FUNCTIONS = [
    "branin"
]

def function_factory(function_name: str) -> Callable:
    """
    TODO Finish Documentation
    """

    raise ValueError(f"Function {function_name} not found")