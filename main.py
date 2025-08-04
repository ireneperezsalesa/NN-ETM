"""
Copyright (C) 2024 Irene Perez-Salesa <i.perez at unizar dot es> (University of Zaragoza)
For more information see <https://github.com/ireneperezsalesa/NN-ETM/blob/master/README.md>

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
from utils.generate_data import *
from algorithms.consensus_lin import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scienceplots


if __name__ == '__main__':

    # Checkpoint folder
    path_to_model = './checkpoints/example_model.pth'  # path to trained NN model for NN-ETM

    # Load NN model
    model = torch.load(path_to_model)
    model.eval()

    # Reference signals
    h = 1e-3
    T = 2
    t_break = 0.8 # randomly changes the reference signals at this time to check how the event threshold adapts accordingly
    N = 3
    Adj = load_graph(N)
    batch_size = 1  # change to test on more than 1 sequence
    # Generate reference signals:
    times_batch, u_batch, u_avg_batch = generate_signals_break(N, h, T, t_break, batch_size)

    # Or load a pre-generated batch of sequences:
    # seq_batch = torch.load('./data/random_20_seqs.pt')
    # batch_size = seq_batch['batch_size']
    # times_batch = seq_batch['t']
    # u_batch = seq_batch['u']
    # u_avg_batch = seq_batch['u_avg']

    # Parameters for NN-ETM
    eps = 1e-3
    sigma = 1

    # Consensus gain
    kappa = 50

    # Initial condition for auxiliary variables. Must fulfill sum(p0)=0
    p0 = 10 * torch.randn(N)
    p0[0] = - torch.sum(p0[1:])
    z0 = 10 * torch.randn(N)

    for i in range(0, batch_size):
        # Select one simulation from the batch
        times = times_batch[i]
        u = u_batch[i]
        u_avg = u_avg_batch[i]

        # Run estimation
        zn, evs, deltas, ei, disag = lin_consensus_test(times, u, Adj, kappa, sigma, eps, model, z0, p0)

        evs_over_time = torch.zeros(times.shape[0], N)
        for t in range(0, times.shape[0]):
            for n in range(0, N):
                evs_over_time[t, n] = sum(evs[0:t, n])

        # Plot estimation results
        plt.style.use(['science', 'ieee'])
        plt.rcParams['text.usetex'] = True
        plt.rcParams['axes.labelsize'] = 'large'
        colores = []
        for color in mcolors.TABLEAU_COLORS:
            colores.append(color)

        fig, axs = plt.subplots(5, 1)
        fig.tight_layout(pad=0)
        fig.set_figheight(3.6)

        for n in range(N):
            axs[0].plot(times.detach(), zn[:, n].detach(), linestyle='solid', c=colores[n])
            axs[1].plot(times.detach(), deltas[:, n].detach(), linestyle='solid', c=colores[n])
            axs[2].plot(times.detach(), ei[:, n].detach(), linestyle='solid', c=colores[n])
            axs[3].plot(times.detach(), disag[:, n].detach(), linestyle='solid', c=colores[n])
            axs[4].plot(times.detach(), evs_over_time[:, n].detach(), linestyle='solid', c=colores[n])

        axs[0].plot(times.detach(), u_avg.detach(), color='k', linestyle='dashed')
        axs[0].set(ylabel='$z_i(t)$')
        axs[0].set_xlim([0, times[-1]])
        axs[1].set(ylabel='$\eta_i(t)$')
        axs[1].set_xlim([0, times[-1]])
        axs[1].set_ylim([0, 1])
        axs[2].set(ylabel='$|u_i(t)|$')
        axs[2].set_xlim([0, times[-1]])
        axs[2].set_ylim([0, 2.5])
        axs[3].set(ylabel='$|d_i(t)|$')
        axs[3].set_xlim([0, times[-1]])
        axs[3].set_ylim([0, 5])
        axs[4].set(xlabel='$t$', ylabel='Events')
        axs[4].set_xlim([0, times[-1]])

        fig.savefig('./figs/example_figure.pdf')

        # Compute communication rate and MSE
        print('Communication rate (normalized between 0 and 1): ', sum(sum(evs)) / (N * times.shape[0]))
        gt_states = u_avg
        estim_states = zn
        err = 0
        for node in range(0, N):
            for time in range(0, times.shape[0]):
                err = err + (torch.linalg.norm(gt_states[time] - estim_states[time, node])
                             * torch.linalg.norm(gt_states[time] - estim_states[time, node]))
        mserr = err / (N * times.shape[0])
        print('MSE: ', mserr)



