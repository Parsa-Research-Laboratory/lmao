import os, sys
import glob
import zipfile
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# import slayer from lava-dl
import lava.lib.dl.slayer as slayer
from lmaao.test_functions.nmnist.nmnist import augment, NMNISTDataset
from lmaao.test_functions.base.process import BaseFunctionProcess

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

class Network(torch.nn.Module):
    def __init__(self, threshold, current_decay, voltage_decay, tau_grad, scale_grad):
        super(Network, self).__init__()

        neuron_params = {
            'threshold': threshold,
            'current_decay': current_decay,
            'voltage_decay': voltage_decay,
            'tau_grad': tau_grad,
            'scale_grad': scale_grad,
            'requires_grad': False,
        }
        neuron_params_drop = {**neuron_params, 'dropout': slayer.neuron.Dropout(p=0.05), }

        self.blocks = torch.nn.ModuleList([
            slayer.block.cuba.Dense(neuron_params_drop, 34 * 34 * 2, 512, weight_norm=True, delay=True),
            slayer.block.cuba.Dense(neuron_params_drop, 512, 512, weight_norm=True, delay=True),
            slayer.block.cuba.Dense(neuron_params, 512, 10, weight_norm=True),
        ])

    def forward(self, spike):
        for block in self.blocks:
            spike = block(spike)
        return spike

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



class NmnistProcess(BaseFunctionProcess):
    '''
    Process for running nmnist with SNN. From lava tutorial:
    https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/nmnist/train.ipynb
    '''

    def __init__(self, **kwargs):
        super().__init__(num_params=5, **kwargs)


@implements(proc=NmnistProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyNmnistProcessModel(PyLoihiProcessModel):
    input_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    output_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float32)

    num_params = LavaPyType(int, int)
    num_outputs = LavaPyType(int, int)

    def train_net(self, threshold, current_decay, voltage_decay, tau_grad, scale_grad, epochs=10):
        net = Network(threshold, current_decay, voltage_decay, tau_grad, scale_grad)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

        data_path = "./lbo/test_functions/nmnist/data"
        training_set = NMNISTDataset(path=data_path, train=True, transform=augment)
        testing_set = NMNISTDataset(path=data_path, train=False)

        train_loader = DataLoader(dataset=training_set, batch_size=32, shuffle=True)
        test_loader = DataLoader(dataset=testing_set, batch_size=32, shuffle=True)

        error = slayer.loss.SpikeRate(true_rate=0.2, false_rate=0.03, reduction='sum')

        stats = slayer.utils.LearningStats()
        assistant = slayer.utils.Assistant(net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict)

        for epoch in range(epochs):
            for i, (input, label) in enumerate(train_loader):  # training loop
                output = assistant.train(input, label)
            print(f'\r[Epoch {epoch:2d}/{epochs}] {stats}', end='')

            for i, (input, label) in enumerate(test_loader):  # training loop
                output = assistant.test(input, label)
            print(f'\r[Epoch {epoch:2d}/{epochs}] {stats}', end='')


            stats.update()

        return stats.testing.best_loss

    def run_spk(self):
        if self.input_port.probe():
            input_data = self.input_port.recv()

            output_packet = np.zeros((self.num_outputs + self.num_params), dtype=np.float32)
            output_packet[:self.num_params] = input_data

            threshold= input_data[0]
            current_decay = input_data[1]
            voltage_decay = input_data[2]
            tau_grad = input_data[3]
            scale_grad = input_data[4]

            best_loss = self.train_net(threshold, current_decay, voltage_decay, tau_grad, scale_grad)
            output_packet[self.num_params:] = best_loss

            self.output_port.send(output_packet)
        else:
            pass
