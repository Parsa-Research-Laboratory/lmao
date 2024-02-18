import argparse
from omegaconf import DictConfig
import os

from test_functions import function_factory

DESCRIPTION = "Hyperdimensional Bayesian Optimization in Lava"

def get_config() -> DictConfig:
    """
    TODO Finish Documentation
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    args = parser.parse_args()

    config: dict = vars(args)
    config: DictConfig = DictConfig(config)

    return config


def main():
    """
    TODO Finish Documentation
    """
    pass

if __name__ == "__main__":
    config = get_config()
    main(config)