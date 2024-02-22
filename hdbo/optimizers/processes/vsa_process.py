from .base_optimizer import BaseOptimizerProcess

class VSAOptimizerProcess(BaseOptimizerProcess):
    """
    Represents a process for the VSA optimizer.

    This class provides an implementation for the VSA optimizer process.

    Attributes:
        num_params (int): The number of parameters.
        num_repeats (int): The number of repeats.
        num_outputs (int): The number of outputs.
        input (InPort): The input port.
        output (OutPort): The output port.
    """

    def __init__(self, num_params: int, num_repeats: int = 1,
                 num_outputs: int = 1, **kwargs):
        """
        Initialize a VSAOptimizerProcess object.

        Args:
            num_params (int): The number of parameters.
            num_repeats (int, optional): The number of repeats. Defaults to 1.
            num_outputs (int, optional): The number of outputs. Defaults to 1.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(num_params, num_repeats, num_outputs, **kwargs)
