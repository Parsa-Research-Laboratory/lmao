from omegaconf import DictConfig
from skopt.space import Space

from .base_optimizer import BaseOptimizerProcess

class VSAOptimizerProcess(BaseOptimizerProcess):
    """
    Represents a process for the VSA optimizer.

    This class provides an implementation for the VSA optimizer process.

    Attributes:
        TODO Finish Documentation
    """

    def __init__(self, config: DictConfig, search_space: Space, **kwargs):
        """
        Initialize a VSAOptimizerProcess object.

        Args:
            TODO Finish Documentation
        """

        assert isinstance(config, DictConfig), "config must be a DictConfig"
        assert isinstance(search_space, Space), "search_space must be a Space"

        super().__init__(
            num_params=search_space.n_dims,
            num_repeats=config.get("num_repeats", 1),
            num_outputs=config.get("num_outputs", 1),
            **kwargs
        )

        self.config: DictConfig = config
        self.search_space: Space = search_space