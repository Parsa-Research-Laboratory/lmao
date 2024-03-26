import argparse
from omegaconf import DictConfig

from lbo.factory import (
    config_factory,
    function_factory,
    VALID_FUNCTIONS,
    VALID_SOLVERS
)
from lbo.solver import BOSolver

DESCRIPTION = "Hyperdimensional Bayesian Optimization in Lava"


def get_config() -> DictConfig:
    """
    Get the configuration for the optimization process.

    Returns:
        DictConfig: The configuration as a `DictConfig` object.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument(
        "--function",
        type=str,
        choices=VALID_FUNCTIONS,
        default="ackley",
        help="The function to optimize",
    )
    parser.add_argument(
        "--return-lp",
        action="store_true",
        help="Return the Lava process wrapper for the function",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        dest="max_iterations",
        help="The maximum number of iterations to run",
    )
    parser.add_argument(
        "--num_initial_points",
        type=int,
        default=25,
        help="The number of initial points to use for optimizer initialization",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="The number of processes to use for optimization",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="The number of times to repeat the optimization at every iteration",
    )
    parser.add_argument(
        "--optimizer_class",
        type=str,
        default="gp-cpu",
        help="The class of optimizer to use",
        choices=VALID_SOLVERS
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The random seed to use for the optimization process",
    )

    args = parser.parse_args()

    config: dict = vars(args)
    config: DictConfig = DictConfig(config)

    return config


def print_intro(config: DictConfig, delimiter="-", delimiter_width=60):
    """
    Print the introduction section of the program.

    Args:
        config (DictConfig): The configuration object containing program
            settings.
        delimiter (str, optional): The delimiter character to use for the
            separator line. Defaults to "-".
        delimiter_width (int, optional): The width of the separator line.
            Defaults to 60.
    """
    print()
    print(delimiter * delimiter_width)
    print(DESCRIPTION)
    print(delimiter * delimiter_width)
    print(f"Configuration:")
    print(f" - Black Box Function: {config.function}")
    print(f" - Return Lava Process: {config.return_lp}")
    print(f" - Max Iterations: {config.max_iterations}")
    print(f" - Number of Initial Points: {config.num_initial_points}")
    print(f" - Number of Processes: {config.num_processes}")
    print(f" - Number of Repeats: {config.num_repeats}")
    print(f" - Optimizer Class: {config.optimizer_class}")
    print(f" - Random Seed: {config.seed}")
    print(delimiter * delimiter_width)


def main(config: DictConfig):
    """
    Executes the main logic of the program.

    Args:
        config (DictConfig): The configuration object containing program
            settings.
            TODO Add Parameters

    Returns:
        None
    """
    print_intro(config)

    config.optimizer = config_factory(config)
    function_process, search_space, minima = function_factory(config.function, return_lp=config.return_lp)
    solver = BOSolver(config)
    solver.solve(
        ufunc=function_process,
        use_lp=config.return_lp,
        search_space=search_space
    )


if __name__ == "__main__":
    config = get_config()
    main(config)