from lava.magma.core.model.py.model import PyAsyncProcessModel
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
import numpy as np
from omegaconf import DictConfig
from skopt.space import Space

from ..base.process import BaseOptimizerProcess

class VSAOptimizerProcess(BaseOptimizerProcess):
    """
    Represents a process for the VSA optimizer.

    This class provides an implementation for the VSA optimizer process.

    Attributes:
        TODO Finish Documentation
    """

    def __init__(self, config: DictConfig, search_space: Space, **kwargs):
        """
        Initialize a VSAOptimizerProcess object.

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

        self.config: DictConfig = config
        self.search_space: Space = search_space

@implements(proc=VSAOptimizerProcess, protocol=AsyncProtocol)
@requires(CPU)
@tag('floating_pt')
class PyAsyncVSAOptimizerModel(PyAsyncProcessModel):
    input_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    output_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float32)

    num_params = LavaPyType(int, int)
    num_outputs = LavaPyType(int, int)
    num_repeats = LavaPyType(int, int)

    def run_async(self):
        print("Running VSAOptimizerProcess P1")