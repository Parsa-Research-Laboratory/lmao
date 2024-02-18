import argparse
from omegaconf import DictConfig
import os

from test_functions import function_factory, VALID_FUNCTIONS

DESCRIPTION = "Hyperdimensional Bayesian Optimization in Lava"

def get_config() -> DictConfig:
    """
    TODO Finish Documentation
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument(
        "--function",
        type=str,
        choices=VALID_FUNCTIONS,
        default="branin",
        help="The function to optimize",
    )

    args = parser.parse_args()

    config: dict = vars(args)
    config: DictConfig = DictConfig(config)

    return config


def print_intro(config: DictConfig, delimiter="-", delimiter_width=60):
    """
    TODO Finish Documentation
    """
    print(delimiter * delimiter_width)
    print(DESCRIPTION)
    print(delimiter * delimiter_width)
    print(f"Configuration:")
    print(f" - Black Box Function: {config.function}")


def main(config: DictConfig):
    """
    TODO Finish Documentation
    """
    print_intro(config)


if __name__ == "__main__":
    config = get_config()
    main(config)