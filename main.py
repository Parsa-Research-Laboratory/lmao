import argparse
from omegaconf import DictConfig

from hdbo.factory import function_factory, VALID_FUNCTIONS, VALID_SOLVERS
from hdbo.solver import BOSolver

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
        "--max-iter",
        type=int,
        default=100,
        help="The maximum number of iterations to run",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="The number of times to repeat the optimization at every iteration",
    )
    parser.add_argument(
        "--solver_class",
        type=str,
        default="vsa-cpu",
        help="The class of solver to use",
        choices=VALID_SOLVERS
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
    print(delimiter * delimiter_width)
    print(DESCRIPTION)
    print(delimiter * delimiter_width)
    print(f"Configuration:")
    print(f" - Black Box Function: {config.function}")
    print(f" - Max Iterations: {config.max_iter}")
    print(f" - Number of Repeats: {config.num_repeats}")
    print(f" - Solver Class: {config.solver_class}")
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

    function_process, search_space, minima = function_factory(config.function)
    solver = BOSolver(config)
    solver.solve(function_process, search_space, minima)


if __name__ == "__main__":
    config = get_config()
    main(config)