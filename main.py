"""
Copyright (C) 2024 Irene Perez-Salesa <i.perez at unizar dot es> (University of Zaragoza)
For more information see <https://github.com/ireneperezsalesa/NN-ETM/blob/main/README.md>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


# Test for the trained NN-ETM

import torch
from torch import nn
from model.model import *
from utils.generate_data import *
from algorithms.consensus_alg import *
import matplotlib.pyplot as plt
import scienceplots


if __name__ == '__main__':

    # Checkpoint folder
    path_to_model = './checkpoints/m_500.pth'  # path to trained model

    # Load NN models
    model = torch.load(path_to_model)
    model.eval()

    # Generate groundtruth training data from system
    h = 1e-3  # simulation step
    T = 10  # simulation time
    N = 5  # number of agents
    Adj = load_graph(N)  # get adjacency matrix for the graph
    batch_size = 1  # change to simulate more than 1 sequence
    times_batch, u_batch, u_avg_batch = generate_signals(N, h, T, batch_size)

    # Trigger design parameters
    eps = 0.001
    sigma = 0.1

    # Consensus gain
    kappa = 5

    for i in range(0, batch_size):
        # Select one simulation from the batch
        times = times_batch[i]
        u = u_batch[i]
        u_avg = u_avg_batch[i]

        # Run estimation
        zn, evs, deltas = lin_consensus_test(times, u, Adj, kappa, sigma, eps, model)

        # Plot estimation results
        plt.style.use(['science', 'ieee'])
        plt.rcParams['text.usetex'] = True

        fig, axs = plt.subplots(2, 1)
        fig.tight_layout(pad=1)
        fig.set_figheight(3.6)

        axs[0].plot(times.detach(), u_avg.detach(), color='k', linestyle='dashed', label='$z(t)$')
        axs[0].plot(times.detach(), zn[:, 0].detach(), color='b', linestyle='solid', label='$i = 1$')
        axs[0].plot(times.detach(), zn[:, 1].detach(), color='r', linestyle='solid', label='$i = 2$')
        #for i in range(0, times.shape[0]):
        #    if evs[i, 0] == 1:
        #        plt.axvline(times[i], color='k', linestyle='dotted', linewidth=0.3)
        axs[0].plot(times.detach(), zn[:, 2].detach(), color='g', linestyle='solid', label='$i = 3$')
        axs[0].plot(times.detach(), zn[:, 3].detach(), color='y', linestyle='solid', label='$i = 4$')
        axs[0].plot(times.detach(), zn[:, 4].detach(), color='c', linestyle='solid', label='$i = 5$')
        axs[0].set(xlabel='$t$', ylabel='$z_i(t)$')
        axs[0].set_xlim([0, times[-1]])
        legend = axs[0].legend(frameon='true', facecolor='white', framealpha=0.9, loc=1)
        frame = legend.get_frame()
        frame.set_linewidth(0)

        axs[1].plot(times.detach(), deltas[:, 0].detach(), color='b', linestyle='solid', label='$i = 1$')
        axs[1].plot(times.detach(), deltas[:, 1].detach(), color='r', linestyle='solid', label='$i = 2$')
        axs[1].plot(times.detach(), deltas[:, 2].detach(), color='g', linestyle='solid', label='$i = 3$')
        axs[1].plot(times.detach(), deltas[:, 3].detach(), color='y', linestyle='solid', label='$i = 4$')
        axs[1].plot(times.detach(), deltas[:, 4].detach(), color='c', linestyle='solid', label='$i = 5$')
        axs[1].set(xlabel='$t$', ylabel='$\eta_i(t)$')
        axs[1].set_xlim([0, times[-1]])
        fig.savefig('./figs/eta.pdf')

        # Compute communication rate and MSE
        print('model: ', model)
        print('comm: ', sum(sum(evs))/(N*times.shape[0]))
        gt_states = u_avg
        estim_states = zn
        err = 0
        for node in range(0, N):
            for time in range(0, times.shape[0]):
                err = err + (torch.linalg.norm(gt_states[time] - estim_states[time, node])
                             * torch.linalg.norm(gt_states[time] - estim_states[time, node]))
        mserr = err/(N*times.shape[0])
        print('mse: ', mserr)
