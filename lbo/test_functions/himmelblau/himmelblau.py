import numpy as np
from skopt.space import Space, Real

SEARCH_SPACE = Space([
    Real(-6, 6, name="x"),
    Real(-6, 6, name="y"),
])

MINIMA: float = 0.0


def himmelblau_function(x: float, y: float) -> float:
    return (((x**2 + y - 11)**2) + (((x + y**2 - 7)**2)))
