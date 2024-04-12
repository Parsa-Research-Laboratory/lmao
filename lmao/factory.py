from omegaconf import DictConfig
from skopt.space import Space
from typing import Callable

VALID_FUNCTIONS = [
    "ackley",
    "himmelblau",
    "goldsteinprice",
    "pilotnet",
    "nmnist"
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
        from lmao.test_functions.ackley.ackley import SEARCH_SPACE, MINIMA

        if return_lp:
            from lmao.test_functions.ackley.process import AckleyProcess
            return AckleyProcess, SEARCH_SPACE, MINIMA
        else:
            from lmao.test_functions.ackley.ackley import ackley_function
            return ackley_function, SEARCH_SPACE, MINIMA

    elif function_name == "himmelblau":
        from lmao.test_functions.himmelblau.himmelblau import SEARCH_SPACE, MINIMA
        if return_lp:
            from lmao.test_functions.himmelblau.process import HimmelblauProcess
            return HimmelblauProcess, SEARCH_SPACE, MINIMA
        else:
            from lmao.test_functions.himmelblau.himmelblau import himmelblau_function
            return himmelblau_function, SEARCH_SPACE, MINIMA

    elif function_name == "goldsteinprice":
        from lmao.test_functions.goldsteinprice.goldsteinprice import SEARCH_SPACE, MINIMA
        if return_lp:
            from lmao.test_functions.goldsteinprice.process import GoldsteinPriceProcess
            return GoldsteinPriceProcess, SEARCH_SPACE, MINIMA
        else:
            from lmao.test_functions.goldsteinprice.goldsteinprice import goldsteinprice_function
            return goldsteinprice_function, SEARCH_SPACE, MINIMA
    elif function_name == "pilotnet":
        from lmao.test_functions.pilotnet.pilotnet import SEARCH_SPACE,MINIMA
        if return_lp:
            from lmao.test_functions.pilotnet.process import PilotNetProcess
            return PilotNetProcess, SEARCH_SPACE, MINIMA
        else:
            raise Exception("Pilotnet is only available as process for hyperparameter optimization")
    elif function_name == "nmnist":
        from lmao.test_functions.nmnist.nmnist import SEARCH_SPACE,MINIMA
        if return_lp:
            from lmao.test_functions.nmnist.process import NmnistProcess
            return NmnistProcess, SEARCH_SPACE, MINIMA
        else:
            raise Exception("Nmnist is only available as process for hyperparameter optimization")
    else:
        raise ValueError(f"Function {function_name} not found")


def optimizer_factory(optimizer_class: str, optimizer_config: DictConfig,
                      search_space: Space) -> Callable:
    """
    Factory function that creates and returns an optimizer based on the specified optimizer class.

    Args:
        optimizer_class (str): The class of the optimizer to create.
            Valid values are "vsa-cpu" and "gp-cpu".
        optimizer_config (DictConfig): The configuration for the optimizer.
        search_space (Space): The search space for the optimizer.

    Returns:
        Callable: The created optimizer.

    Raises:
        ValueError: If the specified optimizer class is not found.
    """

    if optimizer_class == "gp-cpu":
        from lmao.optimizers.gpr import GPROptimizerProcess
        return GPROptimizerProcess(optimizer_config, search_space)
    else:
        raise ValueError(f"Optimizer {optimizer_class} not found")


def config_factory(config: DictConfig) -> DictConfig:
    """
    Factory function for creating a configuration based on the given input.

    Args:
        config (DictConfig): The input configuration.

    Returns:
        DictConfig: The created configuration.

    Raises:
        ValueError: If the optimizer class is invalid.
    """

    assert isinstance(config, DictConfig), "config must be a DictConfig"
    assert "optimizer_class" in config, "optimizer_class must be in config"

    if config.optimizer_class == "gp-cpu":
        from .optimizers.configs import GPR_BASE_CONFIG
        config.optimizer = GPR_BASE_CONFIG
    else:
        raise ValueError(f"Invalid optimizer class: {config.optimizer_class}")
    
    return config