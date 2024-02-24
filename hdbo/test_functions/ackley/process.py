
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
import numpy as np

from ..base.process import BaseFunctionProcess


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


@implements(proc=AckleyProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyAckleyProcessModel(PyLoihiProcessModel):
    input_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    output_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float32)

    num_params = LavaPyType(int, int)
    num_outputs = LavaPyType(int, int)
    num_repeats = LavaPyType(int, int)

    def run_spk(self):
        """
        TODO Finish Documentation
        """
        print("Running Ackley P1")