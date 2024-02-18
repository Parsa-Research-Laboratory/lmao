from typing import Callable

VALID_FUNCTIONS = [
    "ackley",
]

from typing import Callable

def function_factory(function_name: str, return_lp: bool = True) -> Callable:
    """
    Factory function that returns a specific function, search space, and minima
    based on the given function.

    Args:
        function_name (str): The name of the function to be created.
        return_lp (bool): Flag indicating whether to return a Lava
            process or a regular function. Default is True.

    Returns:
        Callable: A tuple containing the function, search space, and minima.

    Raises:
        ValueError: If the given function name is not found.
    """
    if function_name == "ackley":
        from hdbo.test_functions.py.ackley import SEARCH_SPACE, MINIMA

        if return_lp:
            from hdbo.test_functions.processes import AckleyProcess
            return AckleyProcess, SEARCH_SPACE, MINIMA
        else:
            from hdbo.test_functions.py import ackley_function
            return ackley_function, SEARCH_SPACE, MINIMA

    else:
        raise ValueError(f"Function {function_name} not found")