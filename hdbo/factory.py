from skopt.space import Space
from typing import Callable

VALID_FUNCTIONS = [
    "ackley",
]

VALID_SOLVERS = [
    "vsa-cpu",
    "gp-cpu"
]

def validate_return(function_process, search_space, minima):
    """
    Validate the return of the function_factory function.

    Args:
        function_process (Callable): The function process to validate.
        search_space (Space): The search space to validate.
        minima (float): The minima to validate.

    Raises:
        AssertionError: If the function process is not a callable, if the
            search space is not a Space object, or if the minima is not a float.
    """
    assert callable(function_process), "function_process must be a callable"
    assert isinstance(search_space, Space)
    assert isinstance(minima, float), "minima must be a float"

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