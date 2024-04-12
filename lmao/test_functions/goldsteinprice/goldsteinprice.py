import numpy as np
from skopt.space import Space, Real

SEARCH_SPACE = Space([
    Real(-2, 2, name="x"),
    Real(-2, 2, name="y"),
])

MINIMA: float = 3.0


def goldsteinprice_function(x: float, y: float) -> float:
    return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
                        30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
