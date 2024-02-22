from .base_process import BaseFunctionProcess


class AckleyProcess(BaseFunctionProcess):
    """
    Represents the Ackley function for evaluation.

    This class provides an implementation for evaluating the Ackley function
    with input parameters and producing output parameters. It defines the number
    of input parameters, the number of times to repeat the function evaluation,
    and the number of output parameters.

    Attributes:
        num_params (int): The number of input parameters for the function.
        num_repeats (int): The number of times to repeat function evaluation.
        num_outputs (int): The number of output parameters for the function.
        input (InPort): The input port for the function.
        output (OutPort): The output port for the function.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new instance of the AckleyProcess class.

        Args:
            **kwargs: Additional keyword arguments to be passed to the parent.
        """
        super().__init__(num_params=2, **kwargs)