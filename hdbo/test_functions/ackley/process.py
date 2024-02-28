
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

import numpy as np

from hdbo.test_functions.base.process import BaseFunctionProcess


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
        a (float): The 'a' parameter for the Ackley function.
        b (float): The 'b' parameter for the Ackley function.
        c (float): The 'c' parameter for the Ackley function.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new instance of the AckleyProcess class.

        Args:
            **kwargs: Additional keyword arguments to be passed to the parent.
        """
        super().__init__(num_params=2, **kwargs)

        self.a: float = Var(shape=(1,), init=20)
        self.b: float = Var(shape=(1,), init=0.2)
        self.c: float = Var(shape=(1,), init=2 * np.pi)


@implements(proc=AckleyProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyAckleyProcessModel(PyLoihiProcessModel):
    """
    This class represents a PyAckleyProcessModel, which is a subclass of
    PyLoihiProcessModel. It implements the Ackley function and provides a
    method to run the spiking neural network simulation.

    Attributes:
        input_port (PyInPort): The input port for receiving input data.
        output_port (PyOutPort): The output port for sending output data.
        num_params (int): The number of parameters.
        num_outputs (int): The number of outputs.
        num_repeats (int): The number of repeats.
        a (float): The value of parameter 'a'.
        b (float): The value of parameter 'b'.
        c (float): The value of parameter 'c'.
    """

    input_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    output_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float32)

    num_params = LavaPyType(int, int)
    num_outputs = LavaPyType(int, int)
    num_repeats = LavaPyType(int, int)

    a = LavaPyType(float, float)
    b = LavaPyType(float, float)
    c = LavaPyType(float, float)

    def run_spk(self):
        """
        For more information about the Ackley function, see:
        - ./ackley.py 
        - https://en.wikipedia.org/wiki/Ackley_function
        """
        if self.input_port.probe():
            input_data = self.input_port.recv()

            output_packet = np.zeros((self.num_repeats, self.num_outputs + self.num_params))
            output_packet[:, :self.num_params] = input_data

            for repeat in range(self.num_repeats):
                x0 = input_data[repeat, 0]
                x1 = input_data[repeat, 1]

                y = -self.a * np.exp(-self.b * np.sqrt(0.5 * (x0**2 + x1**2))) \
                    - np.exp(0.5 * (np.cos(self.c * x0) + np.cos(self.c * x1))) \
                    + self.a + np.exp(1)

                output_packet[repeat, self.num_params:] = y

            self.output_port.send(output_packet)
        else:
            pass
