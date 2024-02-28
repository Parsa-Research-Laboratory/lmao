from lava.magma.core.model.py.model import PyAsyncProcessModel
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
import numpy as np
from omegaconf import DictConfig
from skopt import Optimizer
from skopt.space import Space
import time

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

        # ------------------------
        # Configuration Parameters
        # ------------------------
        self.max_iterations = Var(shape=(1,), init=config.get("max_iterations", 20))
        self.num_initial_points = Var(shape=(1,), init=config.get("num_initial_points", 5))
        self.seed = Var(shape=(1,), init=config.get("seed", 0))

        # ------------------------
        # Internal State Variables
        # ------------------------
        self.finished = Var(shape=(1,), init=0)
        self.time_step = Var(shape=(1,), init=0)

        # ------------------------
        # Logging Variables
        # ------------------------
        x_log_shape: tuple = (
            self.max_iterations.get(),
            self.num_repeats.get(),
            self.num_params.get()
        )
        self.x_log = Var(
            shape=x_log_shape,
            init=np.zeros(x_log_shape)
        )
        y_log_shape: tuple = (
            self.max_iterations.get(),
            self.num_repeats.get(),
            self.num_outputs.get()
        )
        self.y_log = Var(
            shape=y_log_shape,
            init=np.zeros(y_log_shape)
        )
        time_log_shape: tuple = (self.max_iterations.get(),)
        self.time_log = Var(
            shape=time_log_shape,
            init=np.zeros(time_log_shape)
        )

@implements(proc=GPROptimizerProcess, protocol=AsyncProtocol)
@requires(CPU)
@tag('floating_pt')
class PyAsyncGPROptimizerModel(PyAsyncProcessModel):
    input_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    output_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float32)

    # ------------------------
    # Parent Class Parameters
    # ------------------------
    num_params = LavaPyType(int, int)
    num_outputs = LavaPyType(int, int)
    num_repeats = LavaPyType(int, int)

    # ------------------------
    # Configuration Parameters
    # ------------------------
    max_iterations = LavaPyType(int, int)
    num_initial_points = LavaPyType(int, int)
    seed = LavaPyType(int, int)

    # ------------------------
    # Internal State Variables
    # ------------------------
    finished = LavaPyType(int, int)
    time_step = LavaPyType(int, int)

    # ------------------------
    # Logging Variables
    # ------------------------
    x_log = LavaPyType(np.ndarray, np.float32)
    y_log = LavaPyType(np.ndarray, np.float32)
    time_log = LavaPyType(np.ndarray, np.float32)

    def run_async(self):
        while True:
            if self.check_for_pause_cmd():
                return
            
            if self.check_for_stop_cmd():
                return
            
            # Send initial point to prime the system
            if self.time_step == 0:
                self.acquisition_function: str = "gp_hedge"
                self.acquisition_optimizer: str = "auto"
                self.asking_strategy: str = "cl_min"
                self.base_estimator: str = "GP"
                self.initial_point_estimator: str = "random"
                
                self.optimizer = Optimizer(
                    dimensions=[(-32.0, 32.0)] * self.num_params,
                    acq_func=self.acquisition_function,
                    acq_optimizer=self.acquisition_optimizer,
                    base_estimator=self.base_estimator,
                    initial_point_generator=self.initial_point_estimator,
                    n_initial_points=self.num_initial_points,
                    random_state=self.seed
                )

                output_data: list = self.optimizer.ask(n_points=self.num_repeats)
                output_data: np.ndarray = np.array(output_data)
                self.output_port.send(output_data)

            if self.time_step < self.max_iterations:
                if self.input_port.probe():
                    start_time: float = time.time()
                    new_data: np.ndarray = self.input_port.recv()

                    self.x_log[self.time_step, :, :] = new_data[:, :self.num_params]
                    self.y_log[self.time_step, :, :] = new_data[:, self.num_params:]

                    x = new_data[:, :self.num_params].tolist()
                    y = new_data[:, self.num_params:].tolist()
                    y = [val[0] for val in y]

                    self.optimizer.tell(x, y)

                    print(f"Best Point (Iteration {self.time_step}): {np.min(self.optimizer.get_result().func_vals)}")

                    output_data: list = self.optimizer.ask(
                        n_points=self.num_repeats,
                        strategy=self.asking_strategy
                    )
                    output_data: np.ndarray = np.array(output_data)
                    self.output_port.send(output_data)

                    self.time_log[self.time_step] = time.time() - start_time
                    self.time_step += 1
            else:
                self.finished = 1