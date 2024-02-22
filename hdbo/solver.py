from omegaconf import DictConfig
from skopt.space import Space

from .test_functions.processes import BaseFunctionProcess


def validate_config(config: DictConfig) -> None:
    """
    Validate the configuration dictionary.

    Args:
        config (DictConfig): The configuration dictionary to validate.

    Raises:
        AssertionError: If the config is not a dictionary, or if any of the
            required keys are missing or have invalid values.

    """
    assert isinstance(config, DictConfig), "config must be a dictionary"

    assert "max_iter" in config, "max_iter must be in config"
    assert isinstance(config.max_iter, int), "max_iter must be an integer"
    assert config.max_iter > 0, "max_iter must be greater than 0"

    assert "num_repeats" in config, "num_repeats must be in config"
    assert isinstance(config.num_repeats, int), "num_repeats must be an integer"
    assert config.num_repeats > 0, "num_repeats must be greater than 0"


class BOSolver:
    """
    TODO Finish Documentation
    """
    def __init__(self, config: DictConfig) -> None:
        """
        TODO Finish Documentation
        """
        validate_config(config)

        self.max_iter: int = config.max_iter
        self.num_repeats: int = config.num_repeats
        self.solver_class: str = config.solver_class

    def solve(self, problem: BaseFunctionProcess, search_space: Space,
              minima: float) -> None:
        """
        TODO Finish Documentation
        """
        pass