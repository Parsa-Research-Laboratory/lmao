import argparse
import copy
import numpy as np
from omegaconf import DictConfig
import pickle as pkl
import time

from lmao.factory import (
    config_factory,
    function_factory,
    VALID_FUNCTIONS,
    VALID_SOLVERS
)
from lmao.solver import BOSolver

DESCRIPTION = "Lava Multi-Agent Optimization (LMAO)"


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
        default=10,
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
        default=7,
        help="The random seed to use for the optimization process",
    )
    parser.add_argument(
        "--run-idx",
        type=int,
        default=0,
        help="The index of the run",
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
    print(f" - Black Box Function:  {config.function}")
    print(f" - Return Lava Process: {config.return_lp}")
    print(f" - Max Iterations:      {config.max_iterations}")
    print(f" - Number of IPs:       {config.num_initial_points}")
    print(f" - Number of Processes: {config.num_processes}")
    print(f" - Number of Repeats:   {config.num_repeats}")
    print(f" - Optimizer Class:     {config.optimizer_class}")
    print(f" - Random Seed:         {config.seed}")
    print(f" - Run Index:           {config.run_idx}")
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

    solve_start = time.time()

    results = solver.solve(
        ufunc=function_process,
        use_lp=config.return_lp,
        search_space=search_space
    )

    # print(results)

    solve_end = time.time()
    total_time = solve_end - solve_start
    print(f"Total Time: {total_time}")
    return total_time, results

if __name__ == "__main__":
    time_log = []
    log: dict = {
        "time_log": [],
        "results_log": []
    }
    num_runs = 3

    config_base = get_config()
    config_base.num_processes = config_base.run_idx

    print(config_base.num_processes)
    for i in range(num_runs):
        config = copy.deepcopy(config_base)
        config.seed = i
        times, results = main(config)

        log["time_log"].append(times)
        log["results_log"].append(results)

    print(f"Number of Runs: {num_runs}")
    print(f"Average Time: {np.mean(time_log)}")
    print(f"Standard Deviation: {np.std(time_log)}")
    print(f"Time Log: {time_log}")

    log_path = f"tmp-d10/{config_base.function}_run{config_base.run_idx}.pkl"

    print(log_path)

    with open(log_path, "wb") as f:
        pkl.dump(log, f)



