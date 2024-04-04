import os
import glob
import zipfile
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

import lava.lib.dl.slayer as slayer
from skopt.space import Space, Real

SEARCH_SPACE = Space([
    Real(1.2, 1.3, name="threshold"),
    Real(0.2, 0.3, name="current_decay"),
    Real(0.02, 0.04, name='voltage_decay'),
    Real(0.02, 0.04, name='tau_grad'),
    Real(2.8, 3.2, name="scale_grad")
])

MINIMA: float = 0.0

#Adapted from intel file: https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/nmnist/nmnist.py

def augment(event):
    x_shift = 4
    y_shift = 4
    theta = 10
    xjitter = np.random.randint(2*x_shift) - x_shift
    yjitter = np.random.randint(2*y_shift) - y_shift
    ajitter = (np.random.rand() - 0.5) * theta / 180 * 3.141592654
    sin_theta = np.sin(ajitter)
    cos_theta = np.cos(ajitter)
    event.x = event.x * cos_theta - event.y * sin_theta + xjitter
    event.y = event.x * sin_theta + event.y * cos_theta + yjitter
    return event

class NMNISTDataset(Dataset):
    """NMNIST dataset method

    Parameters
    ----------
    path : str, optional
        path of dataset root, by default 'data'
    train : bool, optional
        train/test flag, by default True
    sampling_time : int, optional
        sampling time of event data, by default 1
    sample_length : int, optional
        length of sample data, by default 300
    transform : None or lambda or fx-ptr, optional
        transformation method. None means no transform. By default Noney.
    """
    def __init__(
        self, path='data',
        train=True,
        sampling_time=1, sample_length=300,
        transform=None,
    ):
        super(NMNISTDataset, self).__init__()
        self.path = path
        if train:
            data_path = path + '/Train'
        else:
            data_path = path + '/Test'

        #Am I stupid? What is this block even doing? Wouldn't 0 mean its not downloaded?
        #I've basically never used glob before
        # assert len(glob.glob(f'{data_path}/')) == 0, \
        #     f'Dataset does not exist. Either set download=True '\
        #     f'or download it from '\
        #     f'https://www.garrickorchard.com/datasets/n-mnist '\
        #     f'to {data_path}/'

        self.samples = glob.glob(f'{data_path}/*/*.bin')
        self.sampling_time = sampling_time
        self.num_time_bins = int(sample_length/sampling_time)
        self.transform = transform

    def __getitem__(self, i):
        filename = self.samples[i]
        # label = int(filename.split('/')[-2]) #linux
        label = int(filename.split('\\')[-2]) #windows
        event = slayer.io.read_2d_spikes(filename)
        if self.transform is not None:
            event = self.transform(event)
        spike = event.fill_tensor(
                torch.zeros(2, 34, 34, self.num_time_bins),
                sampling_time=self.sampling_time,
            )
        return spike.reshape(-1, self.num_time_bins), label

    def __len__(self):
        return len(self.samples)



