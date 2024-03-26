from lava.magma.core.model.py.model import PyAsyncProcessModel
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.process.process import ProcessParameters
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol

import numpy as np
from omegaconf import DictConfig
import os
from skopt import Optimizer
from skopt.space import Space, Real, Integer
import time

from lbo.optimizers.base import BaseOptimizerProcess


class GPROptimizerProcess(BaseOptimizerProcess):
    """
    A class representing the process of optimizing using Gaussian Process
    Regression (GPR) for optimizing black-box functions.

    This class extends the BaseOptimizerProcess class and provides additional
    functionality specific to GPR optimization.

    Attributes:
        max_iterations (Var): The maximum number of iterations for the
            optimization process.
        num_initial_points (Var): The number of initial points to sample for
            the optimization process.
        seed (Var): The seed value for random number generation.
        finished (Var): A variable indicating whether the optimization process
            has finished.
        time_step (Var): The current time step of the optimization process.
        x_log (Var): A log of the input values used during the optimization
            process.
        y_log (Var): A log of the output values obtained during the
            optimization process.
        time_log (Var): A log of the time steps during the optimization
            process.

    Args:
        config (DictConfig): The configuration for the optimization process.
        search_space (Space): The search space for the optimization process.
        **kwargs: Additional keyword arguments to be passed to the
            BaseOptimizerProcess constructor.
    """

    def __init__(self, config: DictConfig, search_space: Space, **kwargs):
        """
        Initialize a GPROptimizerProcess object.

        Args:
            config (DictConfig): The configuration for the optimization process.
            search_space (Space): The search space for the optimization process.
            **kwargs: Additional keyword arguments to be passed to the
                BaseOptimizerProcess constructor.
        """

        assert isinstance(config, DictConfig), "config must be a DictConfig"
        assert isinstance(search_space, Space), "search_space must be a Space"

        super().__init__(num_params=search_space.n_dims,
                         num_processes=config.get("num_processes", 1),
                         num_repeats=config.get("num_repeats", 1),
                         num_outputs=config.get("num_outputs", 1),
                         **kwargs)

        # ------------------------
        # Configuration Parameters
        # ------------------------
        self.max_iterations = Var(
            shape=(1,),
            init=config.max_iterations
        )
        self.num_initial_points = Var(
            shape=(1,),
            init=config.num_initial_points
        )
        self.seed = Var(shape=(1,), init=config.seed)

        # ------------------------
        # Internal State Variables
        # ------------------------
        self.finished = Var(shape=(1,), init=0)
        self.process_ticker = Var(shape=(1,), init=0)
        self.time_step = Var(shape=(1,), init=0)

        # ------------------------
        # Logging Variables
        # ------------------------
        x_log_shape: tuple = (
            self.max_iterations.get(),
            self.num_params.get()
        )
        self.x_log = Var(
            shape=x_log_shape,
            init=np.zeros(x_log_shape)
        )
        y_log_shape: tuple = (
            self.max_iterations.get(),
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

        # ------------------
        # Search Space
        # ------------------
        search_space_shape: tuple = (search_space.n_dims,3)
        local_search_space: np.ndarray = np.zeros(search_space_shape, dtype=np.float32)

        for i, dim in enumerate(search_space.dimensions):
            if isinstance(dim, Real):
                local_search_space[i, 0] = dim.low
                local_search_space[i, 1] = dim.high
            elif isinstance(dim, Integer):
                local_search_space[i, 0] = dim.low
                local_search_space[i, 1] = dim.high
                local_search_space[i, 2] = 1.0
            else:
                raise ValueError(f"Unsupported dimension type: {type(dim)}")
            
        self.search_space = Var(
            shape=search_space_shape,
            init=local_search_space
        )

@implements(proc=GPROptimizerProcess, protocol=AsyncProtocol)
@requires(CPU)
@tag('floating_pt')
class PyAsyncGPROptimizerModel(PyAsyncProcessModel):
    """
    A class representing a PyAsyncGPROptimizerModel.

    This class is responsible for optimizing a function using Gaussian Process
    Regression (GPR). It inherits from the PyAsyncProcessModel class.

    Attributes:
        input_port (PyInPort): The input port for receiving data.
        output_port (PyOutPort): The output port for sending data.
        num_params (int): The number of parameters in the optimization problem.
        num_outputs (int): The number of outputs in the optimization problem.
        num_repeats (int): The number of times to repeat the optimization
            process.
        max_iterations (int): The maximum number of iterations for the
            optimization process.
        num_initial_points (int): The number of initial points to sample.
        seed (int): The seed for random number generation.
        finished (int): Flag indicating whether the optimization process has
            finished.
        time_step (int): The current time step of the optimization process.
        x_log (np.ndarray): Log of input data.
        y_log (np.ndarray): Log of output data.
        time_log (np.ndarray): Log of time taken for each iteration.
    """

    # ----------------------------------
    # Initialize input and output ports
    # for each unique process
    # ----------------------------------
    num_processes: str = os.environ["LAVA_BO_NUM_PROCESSES"]
    num_processes: int = int(num_processes)
    for i in range(num_processes):
        exec(f"input_port_{i} = LavaPyType(PyInPort.VEC_DENSE, np.float32)")
        exec(f"output_port_{i} = LavaPyType(PyOutPort.VEC_DENSE, np.float32)")

    # ------------------------
    # Parent Class Parameters
    # ------------------------
    num_params = LavaPyType(int, int)
    num_processes = LavaPyType(int, int)
    num_outputs = LavaPyType(int, int)
    num_repeats = LavaPyType(int, int)

    # ------------------------
    # Configuration Parameters
    # ------------------------
    max_iterations = LavaPyType(int, int)
    num_initial_points = LavaPyType(int, int)
    seed = LavaPyType(int, int)
    search_space = LavaPyType(np.ndarray, object)

    # ------------------------
    # Internal State Variables
    # ------------------------
    finished = LavaPyType(int, int)
    process_ticker = LavaPyType(int, int)
    time_step = LavaPyType(int, int)

    # ------------------------
    # Logging Variables
    # ------------------------
    x_log = LavaPyType(np.ndarray, np.float32)
    y_log = LavaPyType(np.ndarray, np.float32)
    time_log = LavaPyType(np.ndarray, np.float32)    

    def __init__(self, proc_params: ProcessParameters, *args, **kwargs):
        """
        TODO Finish Documentation
        """
        super().__init__(*args, **kwargs)
   

    def run_async(self):
        """
        Run the optimization process asynchronously.

        This method continuously runs the optimization process until a pause or
        stop command is received. It sends an initial point to prime the system
        and then iteratively receives new data, updates the optimizer, and
        sends new points to evaluate.
        """
        while True:
            if self.check_for_pause_cmd():
                return
            
            if self.check_for_stop_cmd():
                return
            
            # Send initial point to prime the system
            if self.time_step == 0:

                decoded_search_space = []
                for i in range(self.search_space.shape[0]):
                    dim = self.search_space[i]
                    if dim[2] == 0.0:
                        decoded_search_space.append(Real(dim[0], dim[1]))
                    elif dim[2] == 1.0:
                        decoded_search_space.append(Integer(dim[0], dim[1]))
                    else:
                        raise ValueError(f"Unsupported dimension type: {dim[0]}")

                self.acquisition_function: str = "gp_hedge"
                self.acquisition_optimizer: str = "auto"
                self.asking_strategy: str = "cl_min"
                self.base_estimator: str = "GP"
                self.initial_point_estimator: str = "random"
                
                self.optimizer = Optimizer(
                    dimensions=decoded_search_space,
                    acq_func=self.acquisition_function,
                    acq_optimizer=self.acquisition_optimizer,
                    base_estimator=self.base_estimator,
                    initial_point_generator=self.initial_point_estimator,
                    n_initial_points=self.num_initial_points,
                    random_state=self.seed
                )

                # send an initial point to each of the process
                output_data: list = self.optimizer.ask(
                    n_points=self.num_processes,
                    strategy=self.asking_strategy
                )
                for i in range(self.num_processes):
                    output_port: PyOutPort = eval(f"self.output_port_{i}")
                    output_data: np.ndarray = np.array(output_data[i])                
                    output_port.send(output_data)

            if self.time_step < self.max_iterations:
                input_port: PyInPort = eval(f"self.input_port_{self.process_ticker}")
                output_port: PyOutPort = eval(f"self.output_port_{self.process_ticker}")
                self.process_ticker = (self.process_ticker + 1) % self.num_processes
                if input_port.probe():
                    start_time: float = time.time()
                    new_data: np.ndarray = input_port.recv()

                    self.x_log[self.time_step, :] = new_data[:self.num_params]
                    self.y_log[self.time_step, :] = new_data[self.num_params:]
    
                    x = new_data[:self.num_params].tolist()
                    y = new_data[self.num_params:].item()

                    print(f"Optimizer [{self.time_step}]: {new_data}")

                    self.optimizer.tell(x, y)

                    output_data: list = self.optimizer.ask(
                        strategy=self.asking_strategy
                    )
                    output_data: np.ndarray = np.array(output_data)
                    output_port.send(output_data)

                    self.time_log[self.time_step] = time.time() - start_time
                    self.time_step += 1
            else:
                self.finished = 1
