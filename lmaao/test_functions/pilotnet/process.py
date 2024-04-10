from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

import random
import numpy as np
import matplotlib.pyplot as plt
import h5py

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


import lava.lib.dl.slayer as slayer


from lmaao.test_functions.base.process import BaseFunctionProcess


class PilotNetwork(torch.nn.Module):
    def __init__(self, threshold, tau_grad, scale_grad):
        super(PilotNetwork, self).__init__()

        sdnn_params = {  # sigma-delta neuron parameters
            'threshold': threshold,  # delta unit threshold
            'tau_grad': tau_grad,  # delta unit surrogate gradient relaxation parameter
            'scale_grad': scale_grad,  # delta unit surrogate gradient scale parameter
            'requires_grad': False,  # trainable threshold
            'shared_param': True,  # layer wise threshold
            'activation': F.relu,  # activation function
        }
        sdnn_cnn_params = {  # conv layer has additional mean only batch norm
            **sdnn_params,  # copy all sdnn_params
            'norm': slayer.neuron.norm.MeanOnlyBatchNorm,  # mean only quantized batch normalizaton
        }
        sdnn_dense_params = {  # dense layers have additional dropout units enabled
            **sdnn_cnn_params,  # copy all sdnn_cnn_params
            'dropout': slayer.neuron.Dropout(p=0.2),  # neuron dropout
        }

        self.blocks = torch.nn.ModuleList([  # sequential network blocks
            # delta encoding of the input
            slayer.block.sigma_delta.Input(sdnn_params),
            # convolution layers
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 3, 24, 3, padding=0, stride=2, weight_scale=2,
                                          weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 24, 36, 3, padding=0, stride=2, weight_scale=2,
                                          weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 36, 64, 3, padding=(1, 0), stride=(2, 1), weight_scale=2,
                                          weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 64, 64, 3, padding=0, stride=1, weight_scale=2,
                                          weight_norm=True),
            # flatten layer
            slayer.block.sigma_delta.Flatten(),
            # dense layers
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 64 * 45 * 6, 100, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 100, 50, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 50, 10, weight_scale=2, weight_norm=True),
            # linear readout with sigma decoding of output
            slayer.block.sigma_delta.Output(sdnn_dense_params, 10, 1, weight_scale=2, weight_norm=True)
        ])

    def event_rate_loss(self, x, max_rate=0.01):
        mean_event_rate = torch.mean(torch.abs(x))
        return F.mse_loss(F.relu(mean_event_rate - max_rate), torch.zeros_like(mean_event_rate))

    def forward(self, x):
        count = []
        event_cost = 0
        x = x.unsqueeze(-1)
        for block in self.blocks:
            # forward computation is as simple as calling the blocks in a loop
            x = block(x)
            if hasattr(block, 'neuron'):
                event_cost += self.event_rate_loss(x)
                #this is from the original sample, but torch.abs((x) > 0) errors and does not make sense
                #what would be absolute value of a boolean?
                # count.append(torch.sum(torch.abs((x[..., 1:]) > 0).to(x.dtype)).item())
                count.append(torch.sum(torch.abs(x[..., 1:]) > 0).to(x.dtype).item())

        return x, event_cost, torch.FloatTensor(count).reshape((1, -1)).to(x.device)

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))

class PilotNetProcess(BaseFunctionProcess):
    '''
    Process for running pilotnet with SDNN. From lava tutorial:
    https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/pilotnet/train.ipynb
    '''

    def __init__(self, **kwargs):
        super().__init__(num_params=5, **kwargs)
        self.input_feats: np.ndarray = Var(shape=(1288,66,200,3), init=0)
        self.ground_truth: np.ndarray = Var(shape=(1288,), init=0)
        # self.input_feats = np.load("./hdbo/test_functions/pilotnet/NVIDIA_X.npy")
        # self.ground_truth = np.load("./hdbo/test_functions/pilotnet/NVIDIA_Y.npy")
        # p = np.random.permutation(len(self.input_feats))
        # self.input_feats = self.input_feats[p]
        # self.ground_truth = self.ground_truth[p]




@implements(proc=PilotNetProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyPilotNetProcessModel(PyLoihiProcessModel):
    input_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    output_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float32)

    num_params = LavaPyType(int, int)
    num_outputs = LavaPyType(int, int)
    input_feats = LavaPyType(np.ndarray, float)
    ground_truth = LavaPyType(np.ndarray, float)

    def train_net(self,lr, lam, threshold, tau_grad, scale_grad, epochs=10):
        batch = 8

        self.input_feats = np.load("./lbo/test_functions/pilotnet/NVIDIA_X.npy")
        self.ground_truth = np.load("./lbo/test_functions/pilotnet/NVIDIA_Y.npy")

        tensor_in = torch.Tensor(self.input_feats).permute(0,3,1,2)
        tensor_gt = torch.Tensor(self.ground_truth)
        dataset = TensorDataset(tensor_in, tensor_gt)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch, num_workers=0)

        net = PilotNetwork(threshold, tau_grad, scale_grad)
        optimizer = torch.optim.RAdam(net.parameters(), lr=lr, weight_decay=1e-5)
        steps = [60, 120, 160]
        stats = slayer.utils.LearningStats()
        assistant = slayer.utils.Assistant(
            net=net,
            error = lambda output, target: F.mse_loss(output.flatten(), target.flatten()),
            optimizer=optimizer,
            count_log=True,
            lam=lam
        )

        for epoch in range(epochs):
            if epoch in steps:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10/3

            for ndx, (imgs,gts) in enumerate(dataloader):
                assistant.train(imgs, gts)
                print(f'\r[Epoch {epoch:3d}/{epochs}] {stats}', end='')

            if epoch%50==49: print()
            stats.update()

        print()
        return stats.training.best_loss

    #lr, lam, threshold, tau_grad, scale_grad
    def run_spk(self):
        if self.input_port.probe():
            input_data = self.input_port.recv()

            output_packet = np.zeros((self.num_outputs + self.num_params), dtype=np.float32)
            output_packet[:self.num_params] = input_data

            lr= input_data[0]
            lam = input_data[1]
            threshold = input_data[2]
            tau_grad = input_data[3]
            scale_grad = input_data[4]

            best_loss = self.train_net(lr, lam, threshold, tau_grad, scale_grad)
            output_packet[self.num_params:] = best_loss

            self.output_port.send(output_packet)
        else:
            pass
