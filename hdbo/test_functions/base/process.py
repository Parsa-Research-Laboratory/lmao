from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var


class BaseFunctionProcess(AbstractProcess):
    """
    Represents a base function for evaluation.

    This class provides a base implementation for evaluating a function with
    input parameters and producing output parameters. It defines the number
    of input parameters, the number of times to repeat the function evaluation,
    and the number of output parameters.

    Attributes:
        num_params (int): The number of input parameters for the function.
        num_repeats (int): The number of times to repeat function evaluation.
        num_outputs (int): The number of output parameters for the function.
        input (InPort): The input port for the function.
        output (OutPort): The output port for the function.
    """

    def __init__(self, num_params: int, num_repeats: int = 1,
                num_outputs: int = 1, **kwargs):
        """
        Initializes a new instance of the BaseFunction class.

        Args:
            num_params (int): The number of input parameters for the function.
            num_repeats (int, optional): The number of times to repeat the
                function evaluation. Defaults to 1.
            num_outputs (int, optional): The number of output parameters for
                the function. Defaults to 1.
            **kwargs: Additional keyword arguments to be passed to the parent.
        """
        super().__init__(**kwargs)

        assert isinstance(num_params, int)
        assert isinstance(num_repeats, int)
        assert isinstance(num_outputs, int)
        assert num_params > 0
        assert num_repeats > 0
        assert num_outputs == 1

        self.num_params = Var(shape=(1,), init=num_params)
        self.num_outputs = Var(shape=(1,), init=num_outputs)
        self.num_repeats = Var(shape=(1,), init=num_repeats)

        self.input_port = InPort(shape=(num_repeats, num_params,))
        self.output_port = OutPort(shape=(num_repeats, num_params + num_outputs,))