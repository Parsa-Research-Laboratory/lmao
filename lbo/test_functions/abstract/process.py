from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, ProcessParameters
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
import numpy as np
from skopt.space import Space
from typing import Callable, Tuple

from lbo.test_functions.base.process import BaseFunctionProcess, validate_base_args

class AbstractFunctionProcess(BaseFunctionProcess):
    """
    The AbstractFunctionProcess class represents an abstract process for a function in Lava.
    """

    def __init__(self, num_params: int, num_outputs: int, function: Callable,
                 search_space: Space, **kwargs):
        """
        Initializes the AbstractFunctionProcess object.

        Args:
            function (Callable): The function to be optimized.
            search_space (Space): The search space for the function.
        """

        validate_base_args(num_params, num_outputs)

        assert callable(function), "function must be a callable"
        assert isinstance(search_space, Space), "search_space must be a Space"

        process_params = ProcessParameters(initial_parameters={
            "function": function,
            # search_space: search_space
        })

        super().__init__(
            num_params=num_params,
            num_outputs=num_outputs,
            process_params=process_params,
            **kwargs
        )


@implements(proc=AbstractFunctionProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyAbstractFunctionProcessModel(PyLoihiProcessModel):
    input_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    output_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float32)

    num_params = LavaPyType(int, int)
    num_outputs = LavaPyType(int, int)

    def __init__(self, proc_params: ProcessParameters, **kwargs):
        """
        Initialize the Process class.

        Args:
            proc_params (ProcessParameters): The process parameters.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(**kwargs)

        self.user_function: Callable = proc_params["process_params"]["function"]

    def run_spk(self):
        """
        Run the user-defined function.

        This method receives input data from the input port, applies the
        user-defined function to the input data, and sends the output packet
        to the output port.

        Returns:
            None
        """
        if self.input_port.probe():
            input_data = self.input_port.recv()

            y = self.user_function(*input_data).astype(np.float32)

            output_packet = np.zeros((self.num_outputs + self.num_params,), dtype=np.float32)
            output_packet[:self.num_params] = input_data
            output_packet[0] = input_data[0]
            output_packet[1] = input_data[1]
            output_packet[-1] = y

            self.output_port.send(output_packet)


