import numpy as np
from skopt.space import Space, Real

SEARCH_SPACE = Space([
    Real(-32.768, 32.768, name="x"),
    Real(-32.768, 32.768, name="y"),
])

MINIMA: float = 0.0 

def ackley_function(x: float, y: float, a: float = 20, b: float = 0.2,
                    c: float = 2 * np.pi) -> float:
    """
    Calculates the value of the Ackley function for the given input
    coordinates (x, y).

    The Ackley function is a benchmark optimization problem that is commonly
    used to test optimization algorithms. It is defined as:

        f(x, y) = -a * exp(-b * sqrt(0.5 * (x**2 + y**2))) -
            exp(0.5 * (cos(c * x) + cos(c * y))) + a + exp(1)

    where:
    - x, y: Input coordinates
    - a, b, c: Parameters of the function (default values are
        a=20, b=0.2, c=2*pi)

    The function returns the value of the Ackley function for the given
    input coordinates.

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        a (float, optional): The parameter 'a' of the function. Def = 20.
        b (float, optional): The parameter 'b' of the function. Def = 0.2.
        c (float, optional): The parameter 'c' of the function. Def = 2 * pi.

    Returns:
        float: The value of the Ackley function for the given input coordinates.
    """
    return -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 \
        * (np.cos(c * x) + np.cos(c * y))) + a + np.exp(1)