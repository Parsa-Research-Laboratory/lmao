from omegaconf import DictConfig
from skopt.space import Space

from .factory import optimizer_factory
from .optimizers.processes import BaseOptimizerProcess
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
    The BOSolver class represents a solver for optimization problems using
    different forms of Bayesian Optimization in Lava.

    Args:
        config (DictConfig): The configuration for the Solver.

    Attributes:
        max_iter (int): The maximum number of iterations for the solver.
        num_repeats (int): The number of times to repeat the optimization process.
        optimizer_class (str): The class name of the optimizer to use.
        optimizer_config (DictConfig): The configuration for the optimizer.

    Methods:
        solve(function: BaseFunctionProcess, search_space: Space, minima: float) -> None:
            Solves the optimization problem using the specified function, search space, and minimum value.

    TODOs:
        - Ensure the problem process and the optimizer process have the same
            input and output structure.
        - Connect the processes.
        - Run the optimizer.
        - Print the results.
    """
    def __init__(self, config: DictConfig) -> None:
        """
        Initialize the Solver object.

        Args:
            config (DictConfig): The configuration for the Solver.

        Returns:
            None
        """
        validate_config(config)

        self.max_iter: int = config.max_iter
        self.num_repeats: int = config.num_repeats
        self.optimizer_class: str = config.optimizer_class
        self.optimizer_config: DictConfig = config.optimizer

        # Update the optimizer configuration based on the specific
        # configuration for the Solver
        self.optimizer_config.num_repeats = self.num_repeats
        self.optimizer_config.num_outputs = 1

    def solve(self, function: BaseFunctionProcess, search_space: Space,
              minima: float) -> None:
        """
        TODO Finish Documentation
        """

        self.optimizer: BaseOptimizerProcess = optimizer_factory(
            optimizer_class=self.optimizer_class,
            optimizer_config=self.optimizer_config,
            search_space=search_space
        )

        # TODO 1) Ensure the problem process and the optimizer process have the same
        # input and output structure

        # Connect the output of the optimizer to the input of 
        # the function process and vice versa.
        self.optimizer.output_port.connect(function.input_port)
        function.output_port.connect(self.optimizer.input_port)

        # TODO 3) Run the optimizer

        # TODO 4) Print the results