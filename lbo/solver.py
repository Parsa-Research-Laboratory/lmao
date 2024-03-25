from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunContinuous

import os
from omegaconf import DictConfig
from skopt.space import Space
import time
from typing import Union

from .factory import optimizer_factory
from .optimizers.base import BaseOptimizerProcess
from .test_functions.abstract.process import AbstractFunctionProcess
from .test_functions.base.process import BaseFunctionProcess


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

    assert "num_initial_points" in config, "num_initial_points must be in config"
    assert isinstance(config.num_initial_points, int), "num_initial_points must be an integer"
    assert config.num_initial_points > 0, "num_initial_points must be greater than 0"

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
        self.num_initial_points: int = config.num_initial_points
        self.num_repeats: int = config.num_repeats
        self.optimizer_class: str = config.optimizer_class
        self.optimizer_config: DictConfig = config.optimizer

        # Update the optimizer configuration based on the specific
        # configuration for the Solver
        self.optimizer_config.num_repeats = self.num_repeats
        self.optimizer_config.num_outputs = 1

        # ----------------------------------------------------------
        # Variables for ProcessModel Compilation
        #
        # Set the environment variable for the number of processes
        # ----------------------------------------------------------
        os.environ["LAVA_BO_NUM_PROCESSES"] = str(self.optimizer_config.num_processes)

    def solve(self, function: Union[BaseFunctionProcess, callable], search_space: Space,
              ) -> None:
        """
        TODO Finish Documentation
        """

        self.optimizer: BaseOptimizerProcess = optimizer_factory(
            optimizer_class=self.optimizer_class,
            optimizer_config=self.optimizer_config,
            search_space=search_space
        )

        if not isinstance(function, BaseFunctionProcess) and callable(function):
            import inspect
            sig = inspect.signature(function)
            function_process = AbstractFunctionProcess(
                config=self.optimizer_config,
                function=function,
                search_space=search_space
            )
            print(sig.parameters)
            print("Function is callable")
            # exit()
        else:
            function_process = function


        # -------------------------------------------------------
        # For each unique process, connect the unique input and
        # output ports of the optimizer process to the input and
        # output ports of the process
        # -------------------------------------------------------
        num_processes: int = self.optimizer.num_processes.get()
        unique_processes = [function_process() for _ in range(num_processes)]
        for i in range(num_processes):
            optimizer_input_port = eval(f"self.optimizer.input_port_{i}")
            optimizer_output_port = eval(f"self.optimizer.output_port_{i}")
            process = unique_processes[i]

            optimizer_output_port.connect(process.input_port)
            process.output_port.connect(optimizer_input_port)

        finished: bool = False

        while not finished:
            self.optimizer.run(RunContinuous(), Loihi2SimCfg()) 

            time.sleep(1)

            self.optimizer.pause()

            if self.optimizer.finished.get():
                finished = True

        self.optimizer.stop()