from lava.magma.core.model.py.model import PyAsyncProcessModel
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
import numpy as np

from .process import VSAOptimizerProcess

@implements(proc=VSAOptimizerProcess, protocol=AsyncProtocol)
@requires(CPU)
@tag('floating_pt')
class PyAsyncVSAOptimizerProcessModel(PyAsyncProcessModel):
    input_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    output_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float32)

    num_params = LavaPyType(int, int)
    num_outputs = LavaPyType(int, int)
    num_repeats = LavaPyType(int, int)

    def run_async(self):
        print("Running VSAOptimizerProcess P1")