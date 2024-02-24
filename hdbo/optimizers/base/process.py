from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var

class BaseOptimizerProcess(AbstractProcess):
    """
    Base class for optimizer processes.

    This class provides a base implementation for optimizer processes.
    Subclasses can inherit from this class to implement specific processes.

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
        Initialize the BaseOptimizerProcess object.

        Args:
            num_params (int): The number of parameters.
            num_repeats (int, optional): The number of repeats.
                Defaults to 1.
            num_outputs (int, optional): The number of outputs.
                Defaults to 1.
            **kwargs: Additional keyword arguments.

        Raises:
            AssertionError: If any of the input arguments are invalid.

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

        self.input_port = InPort(shape=(num_repeats, num_params + num_outputs,))
        self.output_port = OutPort(shape=(num_repeats, num_params,))
