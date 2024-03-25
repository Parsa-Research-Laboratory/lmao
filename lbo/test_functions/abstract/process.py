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
from omegaconf import DictConfig
from skopt.space import Space
from typing import Callable, Tuple


class AbstractFunctionProcess(AbstractProcess):
    """
    The AbstractFunctionProcess class represents an abstract process for a function in Lava.
    """

    def __init__(self, config: DictConfig, function: Callable, search_space: Space, **kwargs):
        """
        Initializes the AbstractFunctionProcess object.

        Args:
            function (Callable): The function to be optimized.
            search_space (Space): The search space for the function.
        """

        assert isinstance(config, DictConfig), "config must be a DictConfig"
        assert callable(function), "function must be a callable"
        assert isinstance(search_space, Space), "search_space must be a Space"
        assert config.num_params > 0, "num_params must be greater than 0"
        assert config.num_outputs == 1, "num_outputs must be equal to 1"

        process_params = ProcessParameters(initial_parameters={
            function: function,
            # search_space: search_space
        })

        super().__init__(
            process_params=process_params,
            **kwargs
        )

        self.num_params = Var(shape=(1,), init=config.num_params)
        self.num_outputs = Var(shape=(1,), init=config.num_outputs)

        input_shape: Tuple = (self.num_params.get(),)
        output_shape: Tuple = (self.num_params.get() + self.num_outputs.get(),)

        self.input_port = InPort(shape=input_shape)
        self.output_port = OutPort(shape=output_shape)


@implements(proc=AbstractFunctionProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyAbstractFunctionProcessModel(PyLoihiProcessModel):
    input_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    output_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float32)

    num_params = LavaPyType(int, int)
    num_outputs = LavaPyType(int, int)
    num_repeats = LavaPyType(int, int)

    def run_spk(self, process_params: ProcessParameters):
        """
        TODO Finish Documentation
        """
        print(process_params)
        print("Running Abstract Function Process Model")