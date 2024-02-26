from lava.magma.core.model.py.model import PyAsyncProcessModel, PyLoihiProcessModel
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
import numpy as np
from omegaconf import DictConfig
from skopt import Optimizer
from skopt.space import Space

from hdbo.optimizers.base.process import BaseOptimizerProcess


class GPROptimizerProcess(BaseOptimizerProcess):
    """
    Represents a process for the tradition GPR optimizer.

    This class provides an implementation for the traditional optimizer process.

    Attributes:
        TODO Finish Documentation
    """

    def __init__(self, config: DictConfig, search_space: Space, **kwargs):
        """
        Initialize a GPROptimizerProcess object.

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

        self.num_iterations = Var(shape=(1,), init=config.get("max_iterations", 0))
        self.max_iterations = Var(shape=(1,), init=config.get("max_iterations", 100))
        self.finished = Var(shape=(1,), init=0)


@implements(proc=GPROptimizerProcess, protocol=AsyncProtocol)
@requires(CPU)
@tag('floating_pt')
class PyAsyncGPROptimizerModel(PyAsyncProcessModel):
    input_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    output_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float32)

    num_params = LavaPyType(int, int)
    num_outputs = LavaPyType(int, int)
    num_repeats = LavaPyType(int, int)

    num_iterations = LavaPyType(int, int)
    max_iterations = LavaPyType(int, int)
    finished = LavaPyType(int, int)

    def run_async(self):
        while True:
            if self.check_for_pause_cmd():
                return
            
            if self.check_for_stop_cmd():
                return
            
            # Send initial point to prime the system
            if self.num_iterations == 0:
                self.optimizer = Optimizer(
                    dimensions=[(-32.0, 32.0)] * self.num_params,
                    base_estimator="GP",
                    n_initial_points=20,
                    acq_func="EI",
                    acq_optimizer="auto",
                    random_state=0
                )

                output_data: list = self.optimizer.ask(n_points=self.num_repeats)
                output_data: np.ndarray = np.array(output_data)
                self.output_port.send(output_data)

            if self.num_iterations < self.max_iterations:
                if self.input_port.probe():
                    new_data: np.ndarray = self.input_port.recv()

                    x = new_data[:, :self.num_params].tolist()
                    y = new_data[:, self.num_params:].tolist()
                    y = [val[0] for val in y]

                    self.optimizer.tell(x, y)

                    print(f"Best Point (Iteration {self.num_iterations}): {np.min(self.optimizer.get_result().func_vals)}")

                    output_data: list = self.optimizer.ask(n_points=self.num_repeats)
                    output_data: np.ndarray = np.array(output_data)
                    self.output_port.send(output_data)

                    self.num_iterations += 1
            else:
                self.finished = 1